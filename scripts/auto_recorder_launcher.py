#!/usr/bin/env python3
"""
Auto Recorder Launcher Script

This script listens to the environment/save_path topic, deletes the target folder,
starts recording, waits 5 seconds, and then launches the dataset republisher.

Usage:
    python auto_recorder_launcher.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
import subprocess
import shutil
import os
import time
import threading
import signal
import sys
from pathlib import Path


class AutoRecorderLauncher(Node):
    def __init__(self):
        super().__init__('auto_recorder_launcher')
        
        self.get_logger().info('Auto Recorder Launcher - One-shot mode')
        self.save_path = None
        self.done = False
        self.recording_started = False  # Track if recording was started
        self.running_processes = []  # Track running subprocesses
    
    def get_save_path_once(self):
        """Get the save path from the topic once and return"""
        # Subscribe to save_path topic
        subscription = self.create_subscription(
            String,
            'environment/save_path',
            self.save_path_callback,
            10
        )
        
        self.get_logger().info('Waiting for save path from environment/save_path...')
        
        # Spin until we get the message
        while rclpy.ok() and not self.done:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self.save_path
    
    def save_path_callback(self, msg):
        """Callback when save path is received"""
        self.save_path = msg.data
        self.done = True
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
        
        # Reset the done flag
        self.done = False
        self.save_path = old_save_path
        
        # Spin until we get a different save path
        start_time = time.time()
        timeout = 30.0  # 30 second timeout
        
        while rclpy.ok() and (not self.done or self.save_path == old_save_path):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().error('Timeout waiting for new save path')
                raise RuntimeError('Timeout waiting for new save path')
        
        self.get_logger().info(f'New save path received: {self.save_path}')
        return self.save_path
    
    def run_workflow(self):
        """Run the complete workflow once"""
        try:
            # Step 1: Get the initial save path
            save_path = self.get_save_path_once()
            
            if save_path is None:
                self.get_logger().error('Failed to get save path')
                return False
            
            # Step 2: Delete the folder if it exists
            self.delete_folder(save_path)
            
            # Step 3: Reset the environment
            self.reset_environment()
            
            # Step 4: Wait for new save path (different from the old one)
            save_path = self.wait_for_new_save_path(save_path)
            
            # Step 5: Start recording
            self.start_recording()
            
            # Step 6: Launch dataset republisher and eval_lerobot_ros2 in parallel
            self.launch_republisher_and_eval(save_path)
            
            self.get_logger().info('Workflow completed successfully!')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error in auto recorder launcher: {e}')
            return False
    
    def delete_folder(self, folder_path):
        """Delete the folder if it exists"""
        if os.path.exists(folder_path):
            self.get_logger().info(f'Deleting folder: {folder_path}')
            try:
                shutil.rmtree(folder_path)
                self.get_logger().info(f'Successfully deleted: {folder_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to delete folder: {e}')
                raise
        else:
            self.get_logger().info(f'Folder does not exist (will be created by recorder): {folder_path}')
    
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
        """Launch both the dataset republisher and eval_lerobot_ros2 in parallel"""
        self.get_logger().info(f'Launching dataset republisher and eval_lerobot_ros2 with path: {dataset_path}')
        
        try:
            # Build the launch command for dataset republisher
            # Need to properly source the ROS2 workspace with the correct conda environment
            ros_ws_path = os.path.expanduser('/home/baxter/Documents/LeTrack/ros_ws/install/local_setup.zsh')
            
            # The republisher needs to run in the lerobot conda environment, not gr00t
            # We need to deactivate current env and activate lerobot
            republisher_cmd = (
                f'source ~/bin/miniforge/etc/profile.d/conda.sh && '
                f'conda deactivate && '
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
            
            # Build the command for eval_lerobot_ros2
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            eval_script_path = os.path.join(script_dir, 'eval_lerobot_ros2.py')
            
            # You may need to adjust these parameters based on your setup
            eval_cmd = f'python3 {eval_script_path} --wait_for_convergence False --control_frequency 3'
            
            self.get_logger().info(f'Running eval command: {eval_cmd}')
            
            # Launch eval process (non-blocking)
            eval_process = subprocess.Popen(
                eval_cmd,
                shell=True,
                executable='/bin/zsh',
                text=True
            )
            self.running_processes.append(('eval_lerobot_ros2', eval_process))
            
            # Wait for both processes to complete
            self.get_logger().info('Both processes launched. Waiting for completion...')
            
            for name, process in self.running_processes:
                try:
                    returncode = process.wait()
                    self.get_logger().info(f'{name} exited with code: {returncode}')
                    if returncode != 0:
                        self.get_logger().warn(f'{name} exited with non-zero code: {returncode}')
                except KeyboardInterrupt:
                    self.get_logger().info('Process wait interrupted')
                    raise
            
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
        for name, process in self.running_processes:
            if process.poll() is None:
                self.get_logger().info(f'Terminating {name}...')
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.get_logger().warn(f'Force killing {name}...')
                    process.kill()
                    try:
                        process.wait(timeout=1)
                    except:
                        pass
        self.running_processes.clear()


def main(args=None):
    rclpy.init(args=args)
    
    node = AutoRecorderLauncher()
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
