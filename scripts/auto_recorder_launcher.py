#!/usr/bin/env python3
"""
Auto Recorder Launcher Script

This script listens to the environment/save_path topic, deletes the target folder,
starts recording, waits 5 seconds, and then launches the dataset republisher.

Usage:
    python auto_recorder_launcher.py
"""

import rclpy
import rclpy.logging
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool, Trigger
import subprocess
import shutil
import os
import time
import threading
import signal
import sys
from pathlib import Path
import glob
import json
import cv2


class AutoRecorderLauncher(Node):
    def __init__(self, policy_type='groot', policy_path=None, wait_for_convergence='True', control_frequency=3.0, root=None, num_episodes=1, episode_timeout=300.0, real=False):
        super().__init__('auto_recorder_launcher')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)
        
        self.policy_type = policy_type
        self.policy_path = policy_path
        self.wait_for_convergence = wait_for_convergence
        self.control_frequency = control_frequency
        self.root = root
        self.num_episodes = num_episodes
        self.episode_timeout = episode_timeout
        self.real = real
        self.get_logger().info(f'Auto Recorder Launcher - (Policy: {self.policy_type}, Episodes: {self.num_episodes}, Timeout: {self.episode_timeout}s, Real: {self.real})')
        self.save_path = None
        self.done = False
        self.recording_started = False  # Track if recording was started
        self.running_processes = []  # Track running subprocesses
        self.task_completed = False
        self.all_results = {}
        
        # Subscribe to task_completed topic
        self.create_subscription(
            Bool,
            '/task_completed',
            self.task_completed_callback,
            10
        )
        
        # Publisher to reset task_completed
        self.task_completed_pub = self.create_publisher(Bool, '/task_completed', 10)
        
        # Subscribe to save_path topic
        self.create_subscription(
            String,
            'environment/save_path',
            self.save_path_callback,
            10
        )
        
        # Initial cleanup for safety
        self.force_kill_lingering_nodes()
    
    def task_completed_callback(self, msg):
        """Callback when task is completed"""
        if msg.data:
            self.get_logger().info('Task completed signal received!')
            self.task_completed = True

    def get_save_path_once(self):
        """Get the save path from the topic"""
        self.get_logger().info('Waiting for save path from environment/save_path...')
        
        # Spin until we get the message
        while rclpy.ok() and self.save_path is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self.save_path
    
    def save_path_callback(self, msg):
        """Callback when save path is received"""
        if self.save_path != msg.data:
            self.save_path = msg.data
            self.get_logger().info(f'Received save path: {self.save_path}')
    
    def reset_environment(self):
        """Reset the environment via service call"""
        self.get_logger().info('Resetting environment...')
        
        # Create a client for the reset service
        client = self.create_client(Trigger, '/environment/reset')
        
        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Reset service not available')
            raise RuntimeError('Reset service not available')
        
        # Create request
        request = Trigger.Request()
        
        # Call service and wait for response
        future = client.call_async(request)
        
        # Wait for the future to complete with a timeout
        start_time = time.time()
        timeout = 10.0
        
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().error('Reset service call timed out')
                raise RuntimeError('Reset service call timed out')
        
        # Get the result
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Environment reset: {response.message}')
            else:
                self.get_logger().error(f'Failed to reset environment: {response.message}')
                raise RuntimeError(f'Failed to reset environment: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Reset service call failed: {e}')
            raise RuntimeError(f'Reset service call failed: {e}')
    
    def wait_for_new_save_path(self, old_save_path):
        """Wait for the save path to change from old_save_path"""
        self.get_logger().info(f'Waiting for save path to change from: {old_save_path}')
        
        # Spin until we get a different save path
        start_time = time.time()
        timeout = 30.0  # 30 second timeout
        
        while rclpy.ok() and (self.save_path == old_save_path or self.save_path is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().error('Timeout waiting for new save path')
                raise RuntimeError('Timeout waiting for new save path')
        
        self.get_logger().info(f'New save path received: {self.save_path}')
        return self.save_path

    def analyze_trajectory(self, save_path):
        """Run trajectory analysis script"""
        script_path = "/home/baxter/Documents/LeTrack/utils/analyze_ee_trajectory.py"
        if not os.path.exists(script_path):
            self.get_logger().warn(f"Analysis script not found: {script_path}")
            return

        self.get_logger().info(f"Running trajectory visualization for {save_path}")
        try:
            # Run the visualization script with the save path as argument
            cmd = f"python3 {script_path} {save_path}"
            subprocess.run(cmd, shell=True, check=True)
            self.get_logger().info("Trajectory visualization completed")
        except Exception as e:
            self.get_logger().error(f"Failed to run trajectory visualization: {e}")

    def compile_video(self, save_path, episode_num):
        """Compile images into an mp4 video"""
        images_dir = os.path.join(save_path, "scene_camera/rgb")
        if not os.path.exists(images_dir):
            self.get_logger().warn(f"Images directory not found: {images_dir}")
            return

        # Look for png or jpg
        images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        if not images:
            images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
            
        if not images:
            self.get_logger().warn(f"No images found in {images_dir}")
            return

        output_path = os.path.join(save_path, "..", "video", f"task_{episode_num}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.get_logger().info(f"Compiling video to {output_path}")

        try:
            # Use cv2
            first_image = cv2.imread(images[0])
            if first_image is None:
                self.get_logger().error(f"Failed to read first image: {images[0]}")
                return
                
            height, width, layers = first_image.shape
            size = (width, height)
            
            # Try mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 15, size)

            for filename in images:
                img = cv2.imread(filename)
                if img is not None:
                    out.write(img)
            
            out.release()
            self.get_logger().info("Video compilation complete")
        except Exception as e:
            self.get_logger().error(f"Failed to compile video: {e}")

    def aggregate_results(self, save_path, episode_num):
        """Aggregate evaluation results"""
        data = {}
        
        # Load eval_result.json
        result_path = os.path.join(save_path, "eval_result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    data.update(json.load(f))
            except Exception as e:
                self.get_logger().error(f"Failed to load eval_result.json: {e}")
        else:
            self.get_logger().warn(f"Result file not found: {result_path}")

        # Load analysis_results.json
        analysis_path = os.path.join(save_path, "analysis_results.json")
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                    data["analysis"] = analysis_data
            except Exception as e:
                self.get_logger().error(f"Failed to load analysis_results.json: {e}")

        if not data:
            return

        try:
            self.all_results[f"episode_{episode_num}"] = data
            
            # Save combined results in the parent directory
            parent_dir = os.path.dirname(save_path)
            combined_path = os.path.join(parent_dir, "combined_results.json")
            
            with open(combined_path, 'w') as f:
                json.dump(self.all_results, f, indent=4)
                
            self.get_logger().info(f"Results aggregated to {combined_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to aggregate results: {e}")

    def run_workflow(self):
        """Run the complete workflow for num_episodes"""
        for episode in range(self.num_episodes):
            self.get_logger().info(f'Starting episode {episode + 1}/{self.num_episodes}')
            
            # Reset task completed flag
            self.task_completed = False
            
            success = self.run_episode(episode + 1)
            if not success:
                self.get_logger().error(f'Episode {episode + 1} failed')
            
            # Delay between episodes to ensure cleanup
            time.sleep(15.0)
            
        self.get_logger().info('All episodes completed.')
        
        # Post-workflow actions
        if self.save_path:
            # Delete the final save path
            if not self.real:
                self.delete_folder(self.save_path)
            
            # Run video grid script on the parent directory
            parent_dir = os.path.dirname(self.save_path)
            self.run_video_grid_script(parent_dir)
            
        # Shutdown environment
        self.shutdown_environment()
        
        return True

    def run_episode(self, episode_num):
        """Run a single episode"""
        try:
            # Reset task completed signal
            self.task_completed_pub.publish(Bool(data=False))
            
            # Step 1: Get the initial save path
            save_path = self.get_save_path_once()
            
            if save_path is None:
                self.get_logger().error('Failed to get save path')
                return False
            
            # Step 2: Delete the folder if it exists
            if not self.real:
                self.delete_folder(save_path)
            
            # Ensure recording is stopped before resetting
            if not self.real:
                self.stop_recording()

            # Step 3: Reset the environment
            if not self.real:
                self.reset_environment()
            
            # Step 4: Wait for new save path (different from the old one)
            if not self.real:
                save_path = self.wait_for_new_save_path(save_path)
            
            # Step 5: Start recording
            if not self.real:
                self.start_recording()
            
            # Reset task_completed flag to ensure we don't trigger on stale signals
            self.task_completed = False
            
            # Step 6: Launch dataset republisher, cube_swap_node, and eval_lerobot_ros2 in parallel
            self.launch_republisher_and_eval(save_path)
            
            # Step 7: Stop recording
            if not self.real:
                self.stop_recording()
            
            # Step 8: Post-processing
            self.analyze_trajectory(save_path)
            self.compile_video(save_path, episode_num)
            self.aggregate_results(save_path, episode_num)
            
            self.get_logger().info('Episode completed successfully!')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error in episode: {e}')
            self.stop_recording()
            return False
    
    def delete_folder(self, folder_path):
        """Delete scene and wrist camera folders if they exist"""
        for subfolder in ["scene_camera", "wrist_camera"]:
            target_path = os.path.join(folder_path, subfolder)
            if os.path.exists(target_path):
                self.get_logger().info(f'Deleting folder: {target_path}')
                try:
                    shutil.rmtree(target_path)
                    self.get_logger().info(f'Successfully deleted: {target_path}')
                except Exception as e:
                    self.get_logger().error(f'Failed to delete folder: {e}')
                    raise
            else:
                self.get_logger().info(f'Folder does not exist: {target_path}')
    
    def stop_recording(self):
        """Stop recording via service call"""
        if not self.recording_started:
            return
            
        self.get_logger().info('Stopping recording...')
        
        try:
            # Create a client for the recording service
            client = self.create_client(SetBool, '/environment/recording')
            
            # Wait for service to be available
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn('Recording service not available for stopping')
                return
            
            # Create request
            request = SetBool.Request()
            request.data = False
            
            # Call service and wait for response
            future = client.call_async(request)
            
            # Wait for the future to complete with a timeout
            start_time = time.time()
            timeout = 5.0
            
            while not future.done():
                rclpy.spin_once(self, timeout_sec=0.1)
                if time.time() - start_time > timeout:
                    self.get_logger().warn('Stop recording service call timed out')
                    return
            
            # Get the result
            response = future.result()
            if response and response.success:
                self.get_logger().info(f'Recording stopped: {response.message}')
                self.recording_started = False
            else:
                self.get_logger().warn('Failed to stop recording')
                
        except Exception as e:
            self.get_logger().warn(f'Error stopping recording: {e}')
    
    def start_recording(self):
        """Start recording via service call"""
        self.get_logger().info('Starting recording...')
        
        # Create a temporary client for the recording service
        client = self.create_client(SetBool, '/environment/recording')
        
        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Recording service not available')
            raise RuntimeError('Recording service not available')
        
        # Create request
        request = SetBool.Request()
        request.data = True
        
        # Call service and wait for response
        future = client.call_async(request)
        
        # Wait for the future to complete with a timeout, spinning ROS to process callbacks
        start_time = time.time()
        timeout = 10.0
        
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().error('Service call timed out')
                raise RuntimeError('Recording service call timed out')
        
        # Get the result
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Recording started: {response.message}')
                self.recording_started = True  # Mark that recording is active
            else:
                self.get_logger().error(f'Failed to start recording: {response.message}')
                raise RuntimeError(f'Failed to start recording: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            raise RuntimeError(f'Recording service call failed: {e}')
    
    def launch_republisher(self, dataset_path):
        """Launch the dataset republisher with the given dataset path and wait for it to finish"""
        self.get_logger().info(f'Launching dataset republisher with path: {dataset_path}')
        
        try:
            # Build the launch command with proper shell sourcing
            # We need to source the ROS2 workspace and then run the launch command
            ros_ws_path = os.path.expanduser('/home/baxter/Documents/LeTrack/ros_ws/install/local_setup.zsh')
            
            cmd = f'source {ros_ws_path} && ros2 launch so_100_track dataset_republisher.launch.py dataset_path:={dataset_path}'
            
            self.get_logger().info(f'Running command: {cmd}')
            
            # Launch the process and WAIT for it to complete (blocking)
            # This will keep the script running until you manually exit the republisher
            # Use shell=True and executable to run with zsh
            process = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/zsh',
                text=True
            )
            
            self.get_logger().info(f'Dataset republisher exited with code: {process.returncode}')
            
            if process.returncode != 0:
                self.get_logger().warn(f'Dataset republisher exited with non-zero code: {process.returncode}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to launch dataset republisher: {e}')
            raise
    
    def launch_republisher_and_eval(self, dataset_path):
        """Launch dataset republisher, cube_swap_node, and eval_lerobot_ros2 in parallel"""
        self.get_logger().info(f'Launching processes (Real: {self.real})...')
        
        # Determine Conda location
        if os.path.exists("/opt/conda/etc/profile.d/conda.sh"):
            conda_source_cmd = "source /opt/conda/etc/profile.d/conda.sh"
        elif os.environ.get('CONDA_SOURCE_PATH'): # From full_recorder.py
             conda_source_cmd = f"source {os.environ.get('CONDA_SOURCE_PATH')}"
        else:
            conda_source_cmd = "source ~/bin/miniforge/etc/profile.d/conda.sh"

        try:
            if not self.real:
                # Build the launch command for dataset republisher
                # Need to properly source the ROS2 workspace with the correct conda environment
                ros_ws_path = os.path.expanduser('/home/baxter/Documents/LeTrack/ros_ws/install/local_setup.zsh')
                
                # The republisher needs to run in the lerobot conda environment, not gr00t
                # We need to deactivate current env and activate lerobot
                republisher_cmd = (
                    f'{conda_source_cmd} && '
                    f'conda deactivate && '
                    f'export PYTHONPATH="" && '
                    f'conda activate lerobot && '
                    f'source {ros_ws_path} && '
                    f'ros2 launch so_100_track dataset_republisher.launch.py dataset_path:={dataset_path}'
                )
                
                self.get_logger().info(f'Running republisher command: {republisher_cmd}')
                
                # Launch republisher process (non-blocking)
                republisher_process = subprocess.Popen(
                    republisher_cmd,
                    shell=True,
                    executable='/bin/zsh',
                    text=True
                )
                self.running_processes.append(('republisher', republisher_process))
                
                # Wait a moment for republisher to start
                time.sleep(2)
            
            # Build the command for policy evaluation
            # Need to properly source the ROS2 workspace with the correct conda environment
            ros_ws_path = os.path.expanduser('/home/baxter/Documents/LeTrack/ros_ws/install/local_setup.zsh')
            
            if self.policy_type == 'groot':
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                eval_script_path = os.path.join(script_dir, 'eval_lerobot_ros2.py')
                
                # You may need to adjust these parameters based on your setup
                eval_cmd = f'python3 {eval_script_path} --wait_for_convergence {self.wait_for_convergence} --control_frequency {self.control_frequency}'
            
            elif self.policy_type == 'lerobot':
                lerobot_script_path = "/home/baxter/Documents/lerobot/src/lerobot/scripts/lerobot_ros2_control.py"
                
                # Construct command with conda activation for lerobot
                eval_cmd = (
                    f'{conda_source_cmd} && '
                    f'conda deactivate && '
                    f'export PYTHONPATH="" && '
                    f'conda activate lerobot && '
                    f'source {ros_ws_path} && '
                    f'python {lerobot_script_path} --display_data=true'
                )
                
                if self.policy_path:
                    eval_cmd += f' --policy.path={self.policy_path}'
                
                if self.root:
                    eval_cmd += f' --dataset.root={self.root}'
            
            else:
                self.get_logger().error(f'Unknown policy type: {self.policy_type}')
                raise ValueError(f'Unknown policy type: {self.policy_type}')

            self.get_logger().info(f'Running eval command: {eval_cmd}')
            
            # Launch eval process (non-blocking)
            eval_process = subprocess.Popen(
                eval_cmd,
                shell=True,
                executable='/bin/zsh',
                text=True
            )
            self.running_processes.append(('eval_lerobot_ros2', eval_process))
            
            # Wait for both processes to complete OR task_completed signal
            self.get_logger().info('Both processes launched. Waiting for completion or task signal...')
            
            start_time = time.time()

            while True:
                # Check for timeout
                if time.time() - start_time > self.episode_timeout:
                    self.get_logger().error(f'Episode timed out after {self.episode_timeout} seconds!')
                    break

                # Check if task is completed
                if self.task_completed:
                    self.get_logger().info('Task completed signal detected. Stopping processes...')
                    break
                
                # Check if processes are still running
                all_finished = True
                for name, process in self.running_processes:
                    if process.poll() is None:
                        all_finished = False
                        break
                
                if all_finished:
                    self.get_logger().info('All processes finished naturally.')
                    break
                
                # Spin ROS to process callbacks (like task_completed)
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Ensure processes are cleaned up
            self.cleanup_processes()
            
        except KeyboardInterrupt:
            self.get_logger().info('Launch interrupted by user')
            self.cleanup_processes()
            raise
        except Exception as e:
            self.get_logger().error(f'Failed to launch processes: {e}')
            self.cleanup_processes()
            raise
    
    def cleanup_processes(self):
        """Terminate all running subprocesses"""
        # First try to kill the subprocesses we spawned
        for name, process in self.running_processes:
            if process.poll() is None:
                self.get_logger().info(f'Stopping {name} (PID: {process.pid})...')
                
                # Try SIGINT first (Ctrl+C equivalent) - ROS nodes like this
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Try SIGTERM
                    if process.poll() is None:
                        self.get_logger().warn(f'SIGINT timed out for {name}, sending SIGTERM...')
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            # Force kill with SIGKILL
                            if process.poll() is None:
                                self.get_logger().warn(f'SIGTERM timed out for {name}, force killing (SIGKILL)...')
                                process.kill()
                                try:
                                    process.wait(timeout=1)
                                except:
                                    pass
        self.running_processes.clear()
        
        # Aggressive cleanup of any lingering ROS nodes by name
        self.force_kill_lingering_nodes()

    def force_kill_lingering_nodes(self):
        """Force kill known ROS nodes using pkill -9"""
        if self.real:
            # Skip aggressive cleanup on real robot
            return

        nodes_to_kill = [
            "object_pick_place",
            "dataset_republisher",
            "lerobot_ros2_control",
            "eval_lerobot_ros2",
            "cube_swap_node"
        ]
        
        self.get_logger().info("Performing aggressive cleanup of lingering nodes...")
        for node_name in nodes_to_kill:
            try:
                # pkill -9 -f matches the command line pattern
                subprocess.run(f"pkill -9 -f {node_name}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def run_video_grid_script(self, root_path):
        """Run the video grid creation script"""
        # Check standard locations
        possible_paths = [
            "/home/baxter/Documents/LeTrack/utils/create_video_grid.py",
            "/root/Documents/LeTrack/utils/create_video_grid.py",
            os.path.expanduser("~/Documents/LeTrack/utils/create_video_grid.py")
        ]
        
        script_path = None
        for path in possible_paths:
            if os.path.exists(path):
                script_path = path
                break
        
        if not script_path:
            self.get_logger().warn(f"Video grid script not found in locations: {possible_paths}")
            return

        self.get_logger().info(f"Running video grid script at {script_path} for {root_path}")
        try:
            # Run the script with the root path as argument
            cmd = f"python3 {script_path} {root_path}"
            subprocess.run(cmd, shell=True, check=True)
            self.get_logger().info("Video grid creation completed")
        except Exception as e:
            self.get_logger().error(f"Failed to run video grid script: {e}")

    def shutdown_environment(self):
        """Shutdown the environment via service call"""
        self.get_logger().info('Shutting down environment...')
        
        try:
            client = self.create_client(Trigger, '/environment/shutdown')
            
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn('Shutdown service not available')
                return
            
            request = Trigger.Request()
            future = client.call_async(request)
            
            # Wait briefly for response, but don't block long as it might die
            start_time = time.time()
            while not future.done():
                rclpy.spin_once(self, timeout_sec=0.1)
                if time.time() - start_time > 3.0:
                    self.get_logger().info('Shutdown request sent (timed out waiting for response)')
                    return
            
            response = future.result()
            if response.success:
                self.get_logger().info(f'Environment shutdown: {response.message}')
            else:
                self.get_logger().error(f'Failed to shutdown environment: {response.message}')
                
        except Exception as e:
            self.get_logger().error(f'Error shutting down environment: {e}')


def main(args=None):
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto Recorder Launcher')
    parser.add_argument('--policy_type', type=str, default='groot', choices=['groot', 'lerobot'], help='Type of policy to run')
    parser.add_argument('--policy_path', type=str, default=None, help='Path to policy (for lerobot)')
    parser.add_argument('--root', type=str, default=None, help='Root directory for dataset (for lerobot)')
    parser.add_argument('--wait_for_convergence', type=str, default='True', help='Wait for convergence (for groot)')
    parser.add_argument('--control_frequency', type=float, default=3.0, help='Control frequency (for groot)')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--episode_timeout', type=float, default=180.0, help='Episode timeout in seconds')
    parser.add_argument('--real', action='store_true', help='Run in real world mode (no recording, no republisher)')
    
    parsed_args, remaining_args = parser.parse_known_args(args=args)
    
    rclpy.init(args=remaining_args)
    
    node = AutoRecorderLauncher(
        policy_type=parsed_args.policy_type, 
        policy_path=parsed_args.policy_path,
        wait_for_convergence=parsed_args.wait_for_convergence,
        control_frequency=parsed_args.control_frequency,
        root=parsed_args.root,
        num_episodes=parsed_args.num_episodes,
        episode_timeout=parsed_args.episode_timeout,
        real=parsed_args.real
    )
    shutdown_requested = {'value': False}
    
    # Setup signal handler for Ctrl+C
    def signal_handler(sig, frame):
        if shutdown_requested['value']:
            # Already shutting down, force exit
            sys.exit(1)
        shutdown_requested['value'] = True
        node.get_logger().info('Ctrl+C detected - cleaning up...')
        node.cleanup_processes()
        node.stop_recording()
        # Raise KeyboardInterrupt to ensure the main loop exits
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    
    exit_code = 0
    try:
        # Run the workflow once and exit
        success = node.run_workflow()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
        if not shutdown_requested['value']:
            node.cleanup_processes()
            node.stop_recording()
        exit_code = 130
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        if not shutdown_requested['value']:
            node.cleanup_processes()
            node.stop_recording()
        exit_code = 1
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
    
    exit(exit_code)


if __name__ == '__main__':
    main()
