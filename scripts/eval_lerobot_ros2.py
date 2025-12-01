# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ROS2-compatible Gr00T policy eval script for simulation environments.
This script interfaces with ROS2 topics for camera feeds and robot state,
and outputs 3D end effector poses instead of direct robot control.

Example command:

```shell
python eval_lerobot_ros2.py \
    --policy_host=localhost \
    --policy_port=5555 \
    --lang_instruction="Pick up the cube" \
    --camera_topics="['/so101track_cube/camera/rgb/image_raw', '/so101track_cube/wrist_camera/rgb/image_raw']" \
    --camera_keys="['scene_camera', 'wrist_camera']" \
    --robot_state_topic="/so101track_cube/joint_states" \
    --robot_pose_topic="/so101track_cube/right_arm/end_effector/pose" \
    --ee_pose_topic="/right_hand/pose" \
    --gripper_topic="/right_hand/trigger" \
    --action_horizon=8
```

ROS2 topic assumptions:
- Camera topics: sensor_msgs/Image
- Robot state topic: sensor_msgs/JointState
- Robot pose topic: geometry_msgs/Pose
- End effector pose output: geometry_msgs/Pose
- Gripper output: std_msgs/Bool
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Dict, List, Optional

import draccus
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Header, Bool
import cv2

import sys
import os
sys.path.append("/home/baxter/Documents/Isaac-GR00T/gr00t/eval/")
from service import ExternalRobotInferenceClient

#################################################################################


class Gr00tROS2InferenceClient:
    """ROS2-compatible version of the Gr00T inference client.
    
    This class handles the interface between ROS2 data and the Gr00T policy server.
    It converts ROS2 camera images and joint states to the format expected by Gr00T,
    and converts the policy output to end effector poses.
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=None,
        robot_state_keys=None,
        show_images=False,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys or []
        self.robot_state_keys = robot_state_keys or [
            "shoulder_pan", "shoulder_lift", "elbow_flex", 
            "wrist_flex", "wrist_roll", "gripper"
        ]
        self.show_images = show_images
        assert (
            len(self.robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(self.robot_state_keys)}"
        # Update modality keys to match what the policy expects
        # Server expects: state.right_arm_ee_pose, state.right_arm_ee_rot, state.gripper
        self.modality_keys = ["right_arm_ee_pose", "right_arm_ee_rot", "gripper"]

    def get_action(self, camera_images: Dict[str, np.ndarray], 
                   joint_states: np.ndarray, robot_pose: np.ndarray, lang: str) -> List[np.ndarray]:
        """Get action from policy given camera images, joint states, and robot pose.
        
        Args:
            camera_images: Dict mapping camera names to RGB images (H, W, 3)
            joint_states: Array of 6 joint values [5 arm joints + gripper]
            robot_pose: Array of 7 values [x, y, z, qw, qx, qy, qz]
            lang: Language instruction string
            
        Returns:
            List of 6-DOF pose arrays [x, y, z, qw, qx, qy, qz] for each timestep
        """
        print("\n" + "="*80)
        print("DEBUG: Getting action from policy")
        print("="*80)
        
        # Prepare observation dictionary
        obs_dict = {}
        
        # Add camera images
        for key in self.camera_keys:
            if key in camera_images:
                obs_dict[f"video.{key}"] = camera_images[key]
                print(f"DEBUG: Added camera '{key}' - shape: {camera_images[key].shape}, "
                      f"mean pixel value: {camera_images[key].mean():.2f}")
        
        # Show images if requested
        if self.show_images:
            self._view_images(camera_images)

        # Add robot state - use the keys expected by the policy server
        # The server expects: state.right_arm_ee_pose, state.right_arm_ee_rot, state.gripper
        # NOT joint_positions or track_ee_pose/track_ee_rot
        
        # Gripper state
        if len(joint_states) >= 6:
            obs_dict["state.gripper"] = joint_states[5:6].astype(np.float64)  # Gripper state
            print(f"DEBUG: Gripper state: {joint_states[5:6]}")
        else:
            # No gripper data available - raise error
            raise ValueError("Gripper joint data is required but not available")
            
        # Use real robot pose data from the topic if available
        if robot_pose is not None and len(robot_pose) >= 7:
            obs_dict["state.right_arm_ee_pose"] = robot_pose[:3].astype(np.float64)  # [x, y, z]
            obs_dict["state.right_arm_ee_rot"] = robot_pose[3:7].astype(np.float64)  # [qw, qx, qy, qz]
            print(f"DEBUG: Current EE pose: {robot_pose[:3]}")
            print(f"DEBUG: Current EE rotation: {robot_pose[3:7]}")
        else:
            # No placeholder - robot pose is required
            raise ValueError("Robot pose data is required but not available")

        obs_dict["annotation.human.task_description"] = lang

        # Add batch dimension (history = 1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        print(f"DEBUG: Observation dict keys: {list(obs_dict.keys())}")
        
        # Get action chunk from policy
        print("DEBUG: Requesting action from policy server...")
        action_chunk = self.policy.get_action(obs_dict)
        print(f"DEBUG: Received action chunk keys: {list(action_chunk.keys())}")
        
        # Convert to end effector poses
        ee_poses = []
        # Use the right_arm_ee_pose action key
        if "action.right_arm_ee_pose" in action_chunk:
            action_horizon = action_chunk["action.right_arm_ee_pose"].shape[0]
            print(f"DEBUG: Action horizon: {action_horizon}")
            print(f"DEBUG: First action pose: {action_chunk['action.right_arm_ee_pose'][0]}")
            if "action.gripper" in action_chunk:
                print(f"DEBUG: First action gripper: {action_chunk['action.gripper'][0]}")
        else:
            print(f"Warning: Expected action.right_arm_ee_pose not found in action_chunk")
            return []
            
        for i in range(action_horizon):
            pose_data = self._convert_to_ee_pose(action_chunk, i)
            ee_poses.append(pose_data)
        
        print(f"DEBUG: Generated {len(ee_poses)} actions")
        print("="*80 + "\n")
        
        return ee_poses

    def _convert_to_ee_pose(self, action_chunk: Dict[str, np.ndarray], idx: int) -> Dict[str, np.ndarray]:
        """Convert action chunk to end effector pose and gripper data.
        
        Args:
            action_chunk: Policy output dictionary
            idx: Timestep index
            
        Returns:
            Dictionary with 'pose' and 'gripper' keys
        """
        result = {}
        
        # Extract right arm end effector pose
        if "action.right_arm_ee_pose" in action_chunk:
            ee_pos = action_chunk["action.right_arm_ee_pose"][idx]  # [x, y, z]
            result['pose'] = np.array(ee_pos[:3], dtype=np.float64)
        else:
            raise ValueError("action.right_arm_ee_pose not found in action chunk")
            
        # Extract right arm end effector rotation
        if "action.right_arm_ee_rot" in action_chunk:
            ee_rot = action_chunk["action.right_arm_ee_rot"][idx]   # [qw, qx, qy, qz]
            result['rotation'] = np.array(ee_rot[:4], dtype=np.float64)
        else:
            raise ValueError("action.right_arm_ee_rot not found in action chunk")
            
        # Extract gripper value
        if "action.gripper" in action_chunk:
            gripper_val = action_chunk["action.gripper"][idx]
            result['gripper'] = float(gripper_val)
        else:
            raise ValueError("action.gripper not found in action chunk")
            
        return result


    def _view_images(self, camera_images: Dict[str, np.ndarray]):
        """Display camera images for debugging."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
            import matplotlib.pyplot as plt
            
            if len(camera_images) == 1:
                img = list(camera_images.values())[0]
            else:
                # Stack images horizontally
                img = np.concatenate(list(camera_images.values()), axis=1)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title("Camera Views")
            plt.axis("off")
            
            # Save to file instead of displaying
            plt.savefig('/tmp/camera_view.png', bbox_inches='tight', dpi=100)
            plt.close()
            print("Camera view saved to /tmp/camera_view.png")
            
        except ImportError:
            print("matplotlib not available for image display")
        except Exception as e:
            print(f"Error saving camera view: {e}")


#################################################################################


class Gr00tROS2Node(Node):
    """ROS2 node that subscribes to camera and robot state topics,
    runs the Gr00T policy, and publishes end effector pose commands.
    """

    def __init__(self, config):
        super().__init__('gr00t_policy_node')
        
        self.config = config
        
        # Initialize policy client
        self.policy = Gr00tROS2InferenceClient(
            host=config.policy_host,
            port=config.policy_port,
            camera_keys=config.camera_keys,
            show_images=config.show_images,
        )
        
        # Data storage
        self.camera_images = {}
        self.latest_joint_states = None
        self.latest_robot_pose = None
        self.action_buffer = []
        self.action_index = 0
        self.detected_action_horizon = None  # Will be set from first policy response
        
        # Tracking for debugging
        self.last_published_target = None
        self.policy_query_count = 0
        self.last_pose_update_time = None
        self.total_distance_moved = 0.0
        
        # QoS profile
        qos_profile = QoSProfile(depth=10)
        
        # Subscribers
        self._setup_subscribers(qos_profile)
        
        # Publishers
        self.ee_pose_pub = self.create_publisher(
            Pose, 
            config.ee_pose_topic, 
            qos_profile
        )
        
        # Gripper publisher
        self.gripper_pub = self.create_publisher(
            Bool,
            config.gripper_topic,
            qos_profile
        )
        
        # Timer for policy execution
        self.policy_timer = self.create_timer(
            1.0 / config.control_frequency,  # Control frequency in Hz
            self.policy_callback
        )
        
        self.get_logger().info(f"Gr00T ROS2 node initialized")
        self.get_logger().info(f"Policy server: {config.policy_host}:{config.policy_port}")
        self.get_logger().info(f"Language instruction: {config.lang_instruction}")

    def _setup_subscribers(self, qos_profile):
        """Setup ROS2 subscribers for camera and robot state topics."""
        
        # Camera subscribers
        for i, topic in enumerate(self.config.camera_topics):
            camera_key = self.config.camera_keys[i] if i < len(self.config.camera_keys) else f"camera_{i}"
            self.create_subscription(
                Image,
                topic,
                lambda msg, key=camera_key: self.camera_callback(msg, key),
                qos_profile
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic} -> {camera_key}")
        
        # Robot state subscriber
        self.create_subscription(
            JointState,
            self.config.robot_state_topic,
            self.joint_state_callback,
            qos_profile
        )
        self.get_logger().info(f"Subscribed to joint state topic: {self.config.robot_state_topic}")
        
        # Robot pose subscriber
        self.create_subscription(
            PoseStamped,
            self.config.robot_pose_topic,
            self.robot_pose_callback,
            qos_profile
        )
        self.get_logger().info(f"Subscribed to robot pose topic: {self.config.robot_pose_topic}")

    def camera_callback(self, msg: Image, camera_key: str):
        """Callback for camera image messages."""
        try:
            # Convert ROS Image to numpy array directly
            # Assuming RGB8 encoding (3 channels, 8 bits per channel)
            if msg.encoding == 'rgb8':
                # Reshape the data into the correct image format
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = img_array.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                # BGR format - reshape and convert to RGB
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                bgr_image = img_array.reshape((msg.height, msg.width, 3))
                # Convert BGR to RGB
                cv_image = bgr_image[:, :, ::-1]
            elif msg.encoding == 'mono8':
                # Grayscale - convert to RGB by repeating channels
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                gray_image = img_array.reshape((msg.height, msg.width, 1))
                cv_image = np.repeat(gray_image, 3, axis=2)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return
            
            # Check if image has changed
            if camera_key in self.camera_images:
                old_mean = self.camera_images[camera_key].mean()
                new_mean = cv_image.mean()
                if abs(old_mean - new_mean) < 0.1:
                    self.get_logger().debug(f"[CAMERA UPDATE] {camera_key}: UNCHANGED (mean={new_mean:.2f})")
                else:
                    self.get_logger().debug(f"[CAMERA UPDATE] {camera_key}: CHANGED (old_mean={old_mean:.2f}, new_mean={new_mean:.2f})")
            else:
                self.get_logger().debug(f"[CAMERA UPDATE] {camera_key}: FIRST frame (shape={cv_image.shape}, mean={cv_image.mean():.2f})")
                
            self.camera_images[camera_key] = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting camera image: {e}")

    def robot_pose_callback(self, msg: PoseStamped):
        """Callback for robot pose messages."""
        try:
            # Extract position and orientation from Pose message
            pos = msg.pose.position
            orient = msg.pose.orientation

            new_pose = np.array([
                pos.x, pos.y, pos.z,
                orient.w, orient.x, orient.y, orient.z
            ])
            
            # Check if pose has changed and calculate movement speed
            current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
            
            if self.latest_robot_pose is not None:
                pose_delta = np.linalg.norm(new_pose[:3] - self.latest_robot_pose[:3])
                
                # Calculate movement speed
                if self.last_pose_update_time is not None and pose_delta > 0.0001:
                    time_delta = current_time - self.last_pose_update_time
                    if time_delta > 0:
                        speed = pose_delta / time_delta  # meters per second
                        self.total_distance_moved += pose_delta
                        
                        if pose_delta >= 0.001:
                            self.get_logger().debug(
                                f"[POSE UPDATE] CHANGED by {pose_delta:.4f}m - "
                                f"New pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] - "
                                f"Speed: {speed:.4f} m/s"
                            )
                        else:
                            self.get_logger().debug(
                                f"[POSE UPDATE] UNCHANGED (pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}])"
                            )
                    self.last_pose_update_time = current_time
                else:
                    if pose_delta < 0.001:
                        self.get_logger().debug(f"[POSE UPDATE] UNCHANGED (pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}])")
                    else:
                        self.get_logger().debug(f"[POSE UPDATE] CHANGED by {pose_delta:.4f}m - New pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
                        self.last_pose_update_time = current_time
            else:
                self.get_logger().debug(f"[POSE UPDATE] FIRST pose - pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
                self.last_pose_update_time = current_time
            
            # Store as [x, y, z, qw, qx, qy, qz]
            self.latest_robot_pose = new_pose
        except Exception as e:
            self.get_logger().error(f"Error processing robot pose: {e}")

    def joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        try:
            new_joints = np.array(msg.position, dtype=np.float64)
            
            # Check if joints have changed
            if self.latest_joint_states is not None:
                joint_delta = np.linalg.norm(new_joints - self.latest_joint_states)
                if joint_delta < 0.001:
                    self.get_logger().debug(f"[JOINT UPDATE] UNCHANGED (joints={new_joints[:6]})")
                else:
                    self.get_logger().debug(f"[JOINT UPDATE] CHANGED by {joint_delta:.4f} rad - New joints={new_joints[:6]}")
            else:
                self.get_logger().debug(f"[JOINT UPDATE] FIRST joint state - joints={new_joints[:6]}")
            
            # Just store the joint positions directly as numpy array
            self.latest_joint_states = new_joints
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")

    def policy_callback(self):
        """Main policy execution callback."""
        try:
            # Check if we have sufficient data
            if not self._data_ready():
                return
            
            # If we have actions in buffer, execute the next one
            if self.action_buffer and self.action_index < len(self.action_buffer):
                action_dict = self.action_buffer[self.action_index]
                
                # Check if we should wait for convergence
                if self.config.wait_for_convergence and self.last_published_target is not None:
                    current_error = np.linalg.norm(self.last_published_target - self.latest_robot_pose[:3])
                    if current_error > self.config.convergence_threshold:
                        self.get_logger().info(
                            f">>> WAITING for convergence: {current_error*1000:.1f}mm from last target "
                            f"(threshold: {self.config.convergence_threshold*1000:.0f}mm)"
                        )
                        return  # Don't advance to next action yet
                
                self.get_logger().info(f">>> EXECUTING buffered action {self.action_index + 1}/{len(self.action_buffer)}")
                self.get_logger().info(f"    Target pose: [{action_dict['pose'][0]:.3f}, {action_dict['pose'][1]:.3f}, {action_dict['pose'][2]:.3f}]")
                self.get_logger().info(f"    Current pose: [{self.latest_robot_pose[0]:.3f}, {self.latest_robot_pose[1]:.3f}, {self.latest_robot_pose[2]:.3f}]")
                
                # Check if robot pose has changed since last action
                pose_delta = np.linalg.norm(action_dict['pose'] - self.latest_robot_pose[:3])
                self.get_logger().info(f"    Pose delta from current: {pose_delta:.4f}")
                
                # Check movement since last target
                if self.last_published_target is not None:
                    actual_movement = np.linalg.norm(self.latest_robot_pose[:3] - self.last_published_target)
                    if actual_movement < 0.001:
                        self.get_logger().warn(f"    !!! WARNING: Robot barely moved ({actual_movement:.5f}m) since last command !!!")
                
                self._publish_actions(action_dict)
                self.action_index += 1
                return
            
            # Generate new action chunk
            self.policy_query_count += 1
            self.get_logger().info("\n" + "="*80)
            self.get_logger().info(f">>> REQUESTING NEW ACTION CHUNK FROM POLICY (Query #{self.policy_query_count})")
            self.get_logger().info("="*80)
            
            # Robot pose is required
            if self.latest_robot_pose is None:
                self.get_logger().error("Robot pose data is required but not available")
                return
            
            # Check tracking error from last command
            if self.last_published_target is not None:
                tracking_error = np.linalg.norm(self.last_published_target - self.latest_robot_pose[:3])
                self.get_logger().warn(
                    f"!!! TRACKING ERROR: Robot is {tracking_error:.4f}m away from last target !!!"
                )
                self.get_logger().warn(
                    f"    Last target: [{self.last_published_target[0]:.3f}, {self.last_published_target[1]:.3f}, {self.last_published_target[2]:.3f}]"
                )
                self.get_logger().warn(
                    f"    Current pos:  [{self.latest_robot_pose[0]:.3f}, {self.latest_robot_pose[1]:.3f}, {self.latest_robot_pose[2]:.3f}]"
                )
                if tracking_error > 0.01:
                    self.get_logger().error(f"!!! LARGE TRACKING ERROR > 1cm - Robot may not be following commands properly !!!")
                    self.get_logger().error(
                        f"!!! Possible causes: "
                        f"1) Control frequency too high (currently {self.config.control_frequency} Hz), "
                        f"2) Robot controller too slow, "
                        f"3) Commands not reaching robot controller"
                    )
                    
            # Print movement statistics
            if self.total_distance_moved > 0:
                self.get_logger().info(f"Movement stats: Total distance moved: {self.total_distance_moved:.4f}m since start")
            
            # Log current state before requesting action
            self.get_logger().info(f"Current state before policy query:")
            self.get_logger().info(f"  - Gripper: {self.latest_joint_states[5] if len(self.latest_joint_states) > 5 else 'N/A'}")
            self.get_logger().info(f"  - Right arm EE pose: {self.latest_robot_pose[:3]}")
            self.get_logger().info(f"  - Right arm EE rot: {self.latest_robot_pose[3:]}")
            self.get_logger().info(f"  - Available cameras: {list(self.camera_images.keys())}")
            
            action_data = self.policy.get_action(
                self.camera_images.copy(),
                self.latest_joint_states.copy(),
                self.latest_robot_pose.copy(),
                self.config.lang_instruction
            )
            
            # Store action buffer
            self.action_buffer = action_data
            self.action_index = 0
            
            # Detect and validate action horizon
            received_horizon = len(self.action_buffer)
            self.get_logger().info(f">>> Received {received_horizon} actions in new chunk")
            
            if self.detected_action_horizon is None:
                self.detected_action_horizon = received_horizon
                self.get_logger().info(
                    f">>> Detected action horizon from policy: {self.detected_action_horizon}"
                )
                if self.config.action_horizon is not None and self.config.action_horizon != received_horizon:
                    self.get_logger().warn(
                        f"!!! WARNING: Config action_horizon ({self.config.action_horizon}) != "
                        f"received action horizon ({received_horizon}). "
                        f"Using received value ({received_horizon}). "
                        f"To change action horizon, modify the data config on the policy server."
                    )
            elif received_horizon != self.detected_action_horizon:
                self.get_logger().warn(
                    f"!!! WARNING: Received action horizon changed from {self.detected_action_horizon} "
                    f"to {received_horizon}. This is unexpected!"
                )
            
            # Execute first action
            if self.action_buffer:
                action_dict = self.action_buffer[self.action_index]
                self.get_logger().info(f">>> EXECUTING first action from new chunk")
                self.get_logger().info(f"    Target pose: [{action_dict['pose'][0]:.3f}, {action_dict['pose'][1]:.3f}, {action_dict['pose'][2]:.3f}]")
                self._publish_actions(action_dict)
                self.action_index += 1
                
        except Exception as e:
            self.get_logger().error(f"Error in policy callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _data_ready(self) -> bool:
        """Check if we have all necessary data for policy execution."""
        cameras_ready = len(self.camera_images) >= len(self.config.camera_topics)
        joints_ready = self.latest_joint_states is not None
        pose_ready = self.latest_robot_pose is not None
        
        # More verbose logging to debug data availability
        self.get_logger().debug(
            f"Data status: cameras={cameras_ready} ({len(self.camera_images)}/{len(self.config.camera_topics)}), "
            f"joints={joints_ready}, pose={pose_ready}"
        )
        
        if not cameras_ready:
            self.get_logger().info(f"Waiting for camera data. Available cameras: {list(self.camera_images.keys())}")
        if not joints_ready:
            self.get_logger().info("Waiting for joint state data")
        if not pose_ready:
            self.get_logger().info("Waiting for robot pose data - REQUIRED")
            
        # Require ALL data - no placeholders
        return cameras_ready and joints_ready and pose_ready

    def _publish_actions(self, action_dict: Dict[str, np.ndarray]):
        """Publish end effector pose and gripper commands."""
        try:
            # Store this target for tracking error calculation
            self.last_published_target = action_dict['pose'].copy()
            
            # Publish end effector pose
            pose_msg = Pose()
            pose_msg.position = Point(
                x=float(action_dict['pose'][0]), 
                y=float(action_dict['pose'][1]), 
                z=float(action_dict['pose'][2])
            )
            pose_msg.orientation = Quaternion(
                x=float(action_dict['rotation'][1]), 
                y=float(action_dict['rotation'][2]), 
                z=float(action_dict['rotation'][3]), 
                w=float(action_dict['rotation'][0])
            )

            # pose_msg.orientation = Quaternion(
            #     x=0.0, 
            #     y=0.0, 
            #     z=0.0, 
            #     w=1.0
            # )

            self.ee_pose_pub.publish(pose_msg)
            
            # Publish gripper status (boolean based on value threshold)
            gripper_msg = Bool()
            # Convert gripper value to boolean (> 0.5 means closed/True)
            gripper_msg.data = bool(action_dict['gripper'] > 0.5)
            self.gripper_pub.publish(gripper_msg)
            
            self.get_logger().info(
                f"    >>> PUBLISHED: EE pose=[{action_dict['pose'][0]:.3f}, {action_dict['pose'][1]:.3f}, {action_dict['pose'][2]:.3f}], "
                f"rot=[{action_dict['rotation'][0]:.3f}, {action_dict['rotation'][1]:.3f}, {action_dict['rotation'][2]:.3f}, {action_dict['rotation'][3]:.3f}], "
                f"gripper={gripper_msg.data} ({action_dict['gripper']:.3f})"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error publishing actions: {e}")


#################################################################################


@dataclass
class ROS2EvalConfig:
    # Policy server configuration
    policy_host: str = "localhost"
    policy_port: int = 5555
    
    # Task configuration
    lang_instruction: str = "Swap the 2 cubes position using a third location"
    action_horizon: Optional[int] = None  
    """Action horizon (number of actions predicted per policy query). 
    If None, will be automatically detected from the policy server's response.
    The actual action horizon is determined by the policy server's data config (action_indices).
    This parameter is informational - to actually change the action horizon, you need to:
    1. Modify the data config on the policy server (e.g., change action_indices in data_config.py)
    2. Restart the policy server with the new config
    Common values: 8 for So100Track, 16 for other configs."""
    
    # ROS2 topic configuration
    camera_topics: List[str] = None
    camera_keys: List[str] = None
    robot_state_topic: str = "/dataset/joint_states"
    robot_pose_topic: str = "/dataset/right_arm_ee_pose"
    ee_pose_topic: str = "/right_hand/pose"
    gripper_topic: str = "/right_hand/trigger"
    
    # Control configuration
    control_frequency: float = 0.1  # Hz - reduced from 2.0 to give robot time to reach targets
    wait_for_convergence: bool = True  # If True, wait until robot reaches target before next action
    convergence_threshold: float = 0.05  # meters - how close to target before considering "reached"
    
    # Debug options
    show_images: bool = True
    
    def __post_init__(self):
        # Set default camera topics if not provided
        # Based on So100TrackDataConfig: ["video.scene_camera", "video.wrist_camera"]
        if self.camera_topics is None:
            self.camera_topics = [
                "/dataset/scene_camera/rgb",
                "/dataset/wrist_camera/rgb"
            ]
        
        # Set default camera keys if not provided
        if self.camera_keys is None:
            self.camera_keys = ["scene_camera", "wrist_camera"]
        
        # Ensure camera_topics and camera_keys have same length
        if len(self.camera_topics) != len(self.camera_keys):
            raise ValueError(
                f"camera_topics ({len(self.camera_topics)}) and "
                f"camera_keys ({len(self.camera_keys)}) must have same length"
            )


@draccus.wrap()
def main(cfg: ROS2EvalConfig):
    """Main entry point for ROS2 Gr00T evaluation."""
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create and run the node
        node = Gr00tROS2Node(cfg)
        
        node.get_logger().info("Starting Gr00T ROS2 policy evaluation...")
        node.get_logger().info(f"Configuration: {pformat(asdict(cfg))}")
        
        # Spin the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()