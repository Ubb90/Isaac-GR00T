
import argparse
import os
import glob
import yaml
import math
import subprocess
import copy
from datetime import datetime

def generate_bash_script(datasets, wandb_project):
    script_lines = [
        "echo \"--- STARTING GR00T FINETUNE BATCH ---\"",
        "nvidia-smi",
        "git remote remove origin || true",
        "git remote add origin https://github.com/Ubb90/Isaac-GR00T.git",
        "git stash",
        "git pull origin main",
        ""
    ]

    for dataset_path in datasets:
        folder_name = os.path.basename(dataset_path.rstrip('/'))
        # We enforce the container path structure to always be under /mnt/datasets/letrack/
        
        script_to_add = f"""
            # ==========================================
            # DATASET: {folder_name}
            # ==========================================
            DATASET_PATH="/mnt/datasets/letrack/{folder_name}"
            FOLDER_NAME=$(echo "${{DATASET_PATH%/}}" | sed 's/.*\\///; s/_lerobot_v[^/]*$//')

            if [ ! -d "$DATASET_PATH" ]; then
              echo "ERROR: Dataset not found at $DATASET_PATH"
              echo "Listing parent dir to help debug:"
              ls -F "$(dirname "$DATASET_PATH")"
              exit 1
            fi
 
            echo "Dataset found at: $DATASET_PATH"
 
            OUTPUT_DIR="/data/groot/${{FOLDER_NAME}}"
            mkdir -p "$OUTPUT_DIR"

            # Config 1: so100_track
            python scripts/gr00t_finetune.py \\
              --dataset-path "$DATASET_PATH" \\
              --output-dir "${{OUTPUT_DIR}}" \\
              --num-gpus 1 \\
              --max-steps 5000 \\
              --batch-size 48 \\
              --learning_rate 1e-5 \\
              --video-backend torchvision_av \\
              --data-config so100_track\\
              --save-steps 500 \\
              2>&1 | tee "$OUTPUT_DIR/training_log.txt"

            # Config 2: so100_track_medium
            python scripts/gr00t_finetune.py \\
              --dataset-path "$DATASET_PATH" \\
              --output-dir "${{OUTPUT_DIR}}_medium" \\
              --num-gpus 1 \\
              --max-steps 5000 \\
              --batch-size 48 \\
              --learning_rate 1e-5 \\
              --video-backend torchvision_av \\
              --data-config so100_track_medium\\
              --save-steps 500 \\
              2>&1 | tee "$OUTPUT_DIR/training_log_medium.txt"
              
            # Config 3: so100_track_long
            python scripts/gr00t_finetune.py \\
              --dataset-path "$DATASET_PATH" \\
              --output-dir "${{OUTPUT_DIR}}_long" \\
              --num-gpus 1 \\
              --max-steps 5000 \\
              --batch-size 48 \\
              --learning_rate 1e-5 \\
              --video-backend torchvision_av \\
              --data-config so100_track_long\\
              --save-steps 500 \\
              2>&1 | tee "$OUTPUT_DIR/training_log_long.txt"

            # Config 4: so100_track_very_long
            python scripts/gr00t_finetune.py \\
              --dataset-path "$DATASET_PATH" \\
              --output-dir "${{OUTPUT_DIR}}_very_long" \\
              --num-gpus 1 \\
              --max-steps 5000 \\
              --batch-size 48 \\
              --learning_rate 1e-5 \\
              --video-backend torchvision_av \\
              --data-config so100_track_very_long\\
              --save-steps 500 \\
              2>&1 | tee "$OUTPUT_DIR/training_log_very_long.txt"
        """
        script_lines.append(script_to_add)

    return "\n".join(script_lines)

def main():
    parser = argparse.ArgumentParser(description="Generate and run Groot masstune jobs.")
    parser.add_argument("--path", type=str, required=True, help="Base path to search for datasets (e.g., /mnt/datasets/letrack/)")
    parser.add_argument("--robot-name", type=str, required=True, help="Robot name/pattern to match folders (e.g., so101track_cube_swap_moving)")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs/Jobs to split the workload across")
    parser.add_argument("--template", type=str, default="groot-masstune-1.yaml", help="Path to the template YAML file")
    
    args = parser.parse_args()

    # 1. Find datasets
    search_pattern = os.path.join(args.path, f"{args.robot_name}*")
    # We are looking for directories
    found_datasets = sorted([d for d in glob.glob(search_pattern) if os.path.isdir(d)])

    if not found_datasets:
        print(f"No datasets found matching pattern: {search_pattern}")
        # Assuming the user might provide a path that is valid in the container but different locally.
        # But we need to list them, so we assume we can verify them locally.
        dataset_path_check = input("Do you want to continue assuming the datasets exist in the container? (y/n): ")
        if dataset_path_check.lower() != 'y':
            return
        # If we continue here without finding datasets, we can't really split the workload unless user provides list.
        print("Cannot proceed without finding datasets locally to split the workload.")
        return

    print(f"Found {len(found_datasets)} datasets:")
    for d in found_datasets:
        print(f" - {d}")

    # 2. Split workload
    num_jobs = min(args.num_gpus, len(found_datasets))
    chunk_size = math.ceil(len(found_datasets) / num_jobs)
    dataset_chunks = [found_datasets[i:i + chunk_size] for i in range(0, len(found_datasets), chunk_size)]

    print(f"Splitting into {len(dataset_chunks)} jobs.")

    # 3. Load template
    if not os.path.exists(args.template):
        print(f"Template file {args.template} not found.")
        return

    try:
        with open(args.template, 'r') as f:
            template_content = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    # 4. Generate Jobs
    generated_files = []
    
    # Create directory for generated yamls
    os.makedirs("generated_jobs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for i, chunk in enumerate(dataset_chunks):
        job_yaml = copy.deepcopy(template_content)
        
        job_name_suffix = f"-{args.robot_name.replace('_', '-')}-{i+1}-"
        # Truncate if too long (max 63 chars for some k8s fields, but generateName handles suffix)
        if len(job_name_suffix) > 40:
             job_name_suffix = f"-{args.robot_name[:20].replace('_', '-')}-{i+1}-"

        # Update metadata
        if 'metadata' not in job_yaml:
            job_yaml['metadata'] = {}
        
        # We want generateName to be uniqueish prefix
        job_yaml['metadata']['generateName'] = f"groot-finetune{job_name_suffix}"
        
        # Update container spec
        containers = job_yaml['spec']['template']['spec']['containers']
        # Assume the first container is the one we want (named 'trainer' in example)
        container = containers[0]
        
        # Update ENV WANDB_PROJECT
        env_vars = container.get('env', [])
        found_wandb = False
        for env in env_vars:
            if env['name'] == 'WANDB_PROJECT':
                env['value'] = args.robot_name
                found_wandb = True
                break
        if not found_wandb:
            env_vars.append({'name': 'WANDB_PROJECT', 'value': args.robot_name})
        container['env'] = env_vars

        # Update Command/Args
        # The example uses command: ["/bin/bash", "-c"] and args: [script]
        bash_script = generate_bash_script(chunk, args.robot_name)
        container['args'] = [bash_script]

        # Save to file
        output_filename = f"generated_jobs/masstune_{args.robot_name}_{timestamp}_{i+1}.yaml"
        with open(output_filename, 'w') as f:
            yaml.dump(job_yaml, f, sort_keys=False)
        
        generated_files.append(output_filename)

    # 5. Output and Run Commands
    oc_commands = []
    print("\nGenerated Job Files:")
    for gf in generated_files:
        print(f" - {gf}")
        cmd = f"oc create -f {gf}"
        oc_commands.append(cmd)

    print("\nRunning commands:")
    for cmd in oc_commands:
        print(f"Executing: {cmd}")
        # Uncomment to actually run 
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
