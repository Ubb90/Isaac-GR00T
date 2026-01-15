#!/usr/bin/env python3
import subprocess
import time
import os
import signal
import shutil
import sys
import threading
import re
import argparse
from pathlib import Path

# --- Configuration ---
# Detect Conda location
if os.path.exists("/opt/conda/etc/profile.d/conda.sh"):
    CONDA_SOURCE = "source /opt/conda/etc/profile.d/conda.sh"
    # Ensure subprocesses also know where conda is
    os.environ['CONDA_SOURCE_PATH'] = "/opt/conda/etc/profile.d/conda.sh"
else:
    CONDA_SOURCE = "source ~/bin/miniforge/etc/profile.d/conda.sh"
    os.environ['CONDA_SOURCE_PATH'] = os.path.expanduser("~/bin/miniforge/etc/profile.d/conda.sh")

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists("/workspace/LeTrack"):
    # Docker environment
    LETRACK_ROOT = "/workspace/LeTrack"
else:
    # Local environment
    LETRACK_ROOT = os.path.expanduser("~/Documents/LeTrack")

ISAAC_SCRIPT = os.path.join(LETRACK_ROOT, "isaacsim/isaac.sh")
SAVE_DIR_ROOT = "/media/baxter/T7RawData/tmp1"

# List of configurations to run
# Format: (checkpoint_path, data_config)
CONFIGS = [
    # so101track_cube_moving_50 - Checkpoint 500
    ("/media/baxter/storage/models/groot/so101track_cube_swap_moving_1_long/checkpoint-2500", "so100_track"),
]

def infer_data_config(checkpoint_path):
    path = Path(checkpoint_path)
    # Handle trailing slash
    if path.name == "":
        path = path.parent
    
    run_name = path.parent.name
    
    if "very_long" in run_name:
        return "so100_track_very_long"
    elif "medium" in run_name:
        return "so100_track_medium"
    elif "long" in run_name:
        return "so100_track_long"
    else:
        return "so100_track"

def infer_task_name(checkpoint_path):
    path = Path(checkpoint_path)
    # Handle trailing slash
    if path.name == "":
        path = path.parent
    
    run_name = path.parent.name
    
    if "swap" in run_name:
        return "so101track_cube_swap"
    else:
        return "so101track_cube"

def get_output_name(checkpoint_path, data_config):
    """Derives output name from checkpoint path and data config.
    Example: .../so101track_cube_static_reduced/checkpoint-500/ -> so101track_cube_static_reduced_500
    If data_config has medium/long, appends it: so101track_cube_static_reduced_500_medium
    """
    path = Path(checkpoint_path)
    # Handle trailing slash
    if path.name == "":
        path = path.parent
    
    ckpt_name = path.name # checkpoint-500
    run_name = path.parent.name # so101track_cube_static_reduced
    
    # Extract number from checkpoint
    match = re.search(r'checkpoint-(\d+)', ckpt_name)
    if match:
        base_name = f"{run_name}_{match.group(1)}"
    else:
        base_name = f"{run_name}_{ckpt_name}"        
    return base_name

class ProcessRunner:
    def __init__(self):
        self.processes = []
        self.stop_event = threading.Event()

    def run_command(self, cmd, env_name, cwd=None, name="Process"):
        # Clear PYTHONPATH to prevent leakage from the calling environment
        # Force unbuffered output
        
        # Determine correct conda source
        if os.environ.get('CONDA_SOURCE_PATH'):
             conda_src = f"source {os.environ.get('CONDA_SOURCE_PATH')}"
        else:
             conda_src = CONDA_SOURCE

        # Inject specific environment variables and commands based on environment
        pre_cmd = ""
        if env_name == "env_isaacsim":
             pre_cmd = (
                 "export DISPLAY=:99 && "
                 "(Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &) && "
                 "export __NV_PRIME_RENDER_OFFLOAD=1 && "
                 "export __GLX_VENDOR_LIBRARY_NAME=nvidia && "
                "export OMNI_CLIENT_USE_HUB=0 && "
                "export OMNI_USE_HUB=0 && "
                 "export OV_DISABLE_COMPUTE_CACHE=1 && "
             )

        full_cmd = f"{conda_src} && conda deactivate && export PYTHONPATH='' && export PYTHONUNBUFFERED=1 && conda activate {env_name} && {pre_cmd}{cmd}"
        print(f"[{name}] Starting: {full_cmd}")
        
        # Merge current env with unbuffered flag just in case
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['OMNI_KIT_ACCEPT_EULA'] = 'YES'
        
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            executable='/bin/zsh',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            preexec_fn=os.setsid, # Create new process group for easier killing
            env=env
        )
        self.processes.append((name, process))
        return process

    def cleanup(self):
        print("\nCleaning up processes...")
        self.stop_event.set()
        for name, p in self.processes:
            if p.poll() is None:
                print(f"Terminating {name}...")
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        
        # Wait a bit and force kill if needed
        time.sleep(3)
        for name, p in self.processes:
            if p.poll() is None:
                print(f"Killing {name}...")
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        
        # Aggressive cleanup for Isaac Sim
        print("Ensuring Isaac Sim is dead...")
        subprocess.run(f"pkill -u {os.getuid()} -9 -f kit", shell=True)
        
        self.processes = []

def monitor_output(process, name, ready_pattern, ready_event, wait_after_ready=0):
    """Reads output line by line. Sets ready_event when pattern is found."""
    while True:
        line = process.stdout.readline()
        if not line:
            break
        line_str = line.decode('utf-8', errors='replace').strip()
        if line_str:
            print(f"[{name}] {line_str}")
        
        if not ready_event.is_set() and ready_pattern in line_str:
            if wait_after_ready > 0:
                print(f"[{name}] Ready pattern found. Waiting {wait_after_ready}s...")
                time.sleep(wait_after_ready)
            print(f"[{name}] READY SIGNAL RECEIVED")
            ready_event.set()

def run_single_config(ckpt_path, data_config, num_episodes, task_name):
    embodiment_tag = "new_embodiment"
    runner = ProcessRunner()
    
    try:
        # 1. Start Inference Service
        inference_cmd = (
            f"python scripts/inference_service.py "
            f"--model-path {ckpt_path} "
            f"--server "
            f"--embodiment_tag {embodiment_tag} "
            f"--data_config {data_config}"
        )
        
        inf_ready = threading.Event()
        inf_proc = runner.run_command(inference_cmd, "gr00t", cwd=WORKSPACE_ROOT, name="Inference")
        
        inf_thread = threading.Thread(
            target=monitor_output,
            args=(inf_proc, "Inference", "Server is ready and listening on tcp://0.0.0.0:5555", inf_ready)
        )
        inf_thread.daemon = True
        inf_thread.start()

        # 2. Start Isaac Sim
        # Note: Using dynamic paths based on LETRACK_ROOT
        static_flag = "--static " if "static" in ckpt_path else ""
        isaac_cmd = (
            f"{ISAAC_SCRIPT} "
            f"--urdf '{os.path.join(LETRACK_ROOT, 'ros_ws/src/so_100_track/urdf/so_100_arm_wheel.urdf')}' "
            f"--rmp '{os.path.join(LETRACK_ROOT, 'ros_ws/src/so_100_track/config')}' "
            f"-r '{task_name}' "
            f"--fps 3 "
            f"--save-dir '{SAVE_DIR_ROOT}' "
            f"--target-size='640x480' "
            f"--disable-depth "
            f"{static_flag}"
            f"--evaluate "
            "--/app/audio/enabled=false "
            "--portable-root '/media/baxter/T7RawData/isaac_portable' "
            "--vv"
        )
        
        isaac_ready = threading.Event()
        isaac_proc = runner.run_command(isaac_cmd, "env_isaacsim", cwd=LETRACK_ROOT, name="IsaacSim")
        
        # Pattern: "Robot so101track_cube - root_joint: /so_100_arm/root_joint"
        # The prompt mentions two lines, but matching the second unique one is sufficient.
        isaac_thread = threading.Thread(
            target=monitor_output,
            args=(isaac_proc, "IsaacSim", "root_joint: /so_100_arm/root_joint", isaac_ready, 10)
        )
        isaac_thread.daemon = True
        isaac_thread.start()

        # Wait for both to be ready
        print("Waiting for Inference and Isaac Sim to be ready...")
        while not (inf_ready.is_set() and isaac_ready.is_set()):
            if inf_proc.poll() is not None:
                raise RuntimeError("Inference process died unexpectedly")
            if isaac_proc.poll() is not None:
                raise RuntimeError("Isaac Sim process died unexpectedly")
            time.sleep(1)
        
        print("Both services ready! Starting Auto Recorder...")

        # 3. Start Auto Recorder
        recorder_cmd = f"python scripts/auto_recorder_launcher.py --policy-type groot --num_episodes {num_episodes}"
        
        # We run this one blocking (wait for it to finish)
        # But we still need to stream output
        recorder_proc = runner.run_command(recorder_cmd, "gr00t", cwd=WORKSPACE_ROOT, name="Recorder")
        
        # Stream recorder output to console
        while recorder_proc.poll() is None:
            line = recorder_proc.stdout.readline()
            if line:
                print(f"[Recorder] {line.decode('utf-8', errors='replace').strip()}")
        
        print("Auto Recorder finished.")
        
    finally:
        runner.cleanup()

    # 4. Rename folder
    output_name = get_output_name(ckpt_path, data_config)
    src_path = os.path.join(SAVE_DIR_ROOT, task_name)
    dst_path = os.path.join(SAVE_DIR_ROOT, output_name)
    
    if os.path.exists(src_path):
        print(f"Renaming {src_path} to {dst_path}")
        if os.path.exists(dst_path):
            print(f"Warning: Destination {dst_path} already exists. Removing it.")
            shutil.rmtree(dst_path)
        shutil.move(src_path, dst_path)
    else:
        print(f"Warning: Source folder {src_path} not found!")

def main():
    parser = argparse.ArgumentParser(description="Run full recording pipeline")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to run per config")
    parser.add_argument("--config-list", type=str, help="Path to file containing list of checkpoints to run")
    args = parser.parse_args()

    configs_to_run = []
    if args.config_list:
        if not os.path.exists(args.config_list):
            print(f"Error: Config list file {args.config_list} not found.")
            return
            
        print(f"Loading configurations from {args.config_list}...")
        with open(args.config_list, 'r') as f:
            for line in f:
                ckpt_path = line.strip()
                if ckpt_path:
                    data_conf = infer_data_config(ckpt_path)
                    task_name = infer_task_name(ckpt_path)
                    configs_to_run.append((ckpt_path, data_conf, task_name))
        print(f"Loaded {len(configs_to_run)} configurations.")
    else:
        for ckpt, conf in CONFIGS:
            task = infer_task_name(ckpt)
            configs_to_run.append((ckpt, conf, task))

    for ckpt, data_conf, task_name in configs_to_run:
        print(f"\n{'='*50}")
        print(f"Running config: {ckpt}")
        print(f"Data config: {data_conf}")
        print(f"Task name: {task_name}")
        print(f"{'='*50}\n")
        run_single_config(ckpt, data_conf, args.num_episodes, task_name)
        time.sleep(5) # Cooldown between runs

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
