
import argparse
import os
import glob
import math
import subprocess
from datetime import datetime

def generate_bash_script(datasets, wandb_project):
    """Generate bash script for multiple datasets"""
    script_lines = [
        'echo "--- STARTING GR00T FINETUNE BATCH ---"',
        'nvidia-smi',
        'git remote remove origin || true',
        'git remote add origin https://github.com/Ubb90/Isaac-GR00T.git',
        'git stash',
        'git pull origin main',
        ''
    ]

    for dataset_path in datasets:
        folder_name = os.path.basename(dataset_path.rstrip('/'))
        
        script_lines.extend([
            '# ==========================================',
            f'# DATASET: {folder_name}',
            '# ==========================================',
            f'DATASET_PATH="/mnt/datasets/letrack/{folder_name}"',
            'FOLDER_NAME=$(echo "${DATASET_PATH%/}" | sed \'s/.*\\/\\/\\/; s/_lerobot_v[^/]*$//\')',
            '',
            'if [ ! -d "$DATASET_PATH" ]; then',
            '  echo "ERROR: Dataset not found at $DATASET_PATH"',
            '  echo "Listing parent dir to help debug:"',
            '  ls -F "$(dirname "$DATASET_PATH")"',
            '  exit 1',
            'fi',
            '',
            'echo "Dataset found at: $DATASET_PATH"',
            '',
            'OUTPUT_DIR="/data/groot/${FOLDER_NAME}"',
            'mkdir -p "$OUTPUT_DIR"',
            '',
            '# Config 1: so100_track (short horizon)',
            'python scripts/gr00t_finetune.py \\',
            '  --dataset-path "$DATASET_PATH" \\',
            '  --output-dir "${OUTPUT_DIR}" \\',
            '  --num-gpus 1 \\',
            '  --max-steps 5000 \\',
            '  --batch-size 48 \\',
            '  --learning_rate 1e-5 \\',
            '  --video-backend torchvision_av \\',
            '  --data-config so100_track\\',
            '  --save-steps 500 \\',
            '  2>&1 | tee "$OUTPUT_DIR/training_log.txt"',
            '',
            '# Config 2: so100_track_medium',
            'python scripts/gr00t_finetune.py \\',
            '  --dataset-path "$DATASET_PATH" \\',
            '  --output-dir "${OUTPUT_DIR}_medium" \\',
            '  --num-gpus 1 \\',
            '  --max-steps 5000 \\',
            '  --batch-size 48 \\',
            '  --learning_rate 1e-5 \\',
            '  --video-backend torchvision_av \\',
            '  --data-config so100_track_medium\\',
            '  --save-steps 500 \\',
            '  2>&1 | tee "$OUTPUT_DIR/training_log_medium.txt"',
            '',
            '# Config 3: so100_track_long',
            'python scripts/gr00t_finetune.py \\',
            '  --dataset-path "$DATASET_PATH" \\',
            '  --output-dir "${OUTPUT_DIR}_long" \\',
            '  --num-gpus 1 \\',
            '  --max-steps 5000 \\',
            '  --batch-size 48 \\',
            '  --learning_rate 1e-5 \\',
            '  --video-backend torchvision_av \\',
            '  --data-config so100_track_long\\',
            '  --save-steps 500 \\',
            '  2>&1 | tee "$OUTPUT_DIR/training_log_long.txt"',
            '',
            '# Config 4: so100_track_very_long',
            'python scripts/gr00t_finetune.py \\',
            '  --dataset-path "$DATASET_PATH" \\',
            '  --output-dir "${OUTPUT_DIR}_very_long" \\',
            '  --num-gpus 1 \\',
            '  --max-steps 5000 \\',
            '  --batch-size 48 \\',
            '  --learning_rate 1e-5 \\',
            '  --video-backend torchvision_av \\',
            '  --data-config so100_track_very_long\\',
            '  --save-steps 500 \\',
            '  2>&1 | tee "$OUTPUT_DIR/training_log_very_long.txt"',
            ''
        ])

    return '\n'.join(script_lines)

def main():
    parser = argparse.ArgumentParser(description="Generate and run Groot masstune jobs.")
    parser.add_argument("--path", type=str, required=True, help="Base path to search for datasets")
    parser.add_argument("--robot-name", type=str, required=True, help="Robot name/pattern to match folders")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs/Jobs to split workload across")
    parser.add_argument("--template", type=str, default="groot-masstune.yaml", help="Path to template YAML file")
    
    args = parser.parse_args()

    # 1. Find datasets
    search_pattern = os.path.join(args.path, f"{args.robot_name}*")
    found_datasets = sorted([d for d in glob.glob(search_pattern) if os.path.isdir(d)])

    if not found_datasets:
        print(f"No datasets found matching pattern: {search_pattern}")
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

    # Read template as text
    with open(args.template, 'r') as f:
        template_lines = f.readlines()

    # Create directory for generated yamls
    os.makedirs("generated_jobs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 4. Generate Jobs by modifying template text
    generated_files = []
    
    for i, chunk in enumerate(dataset_chunks):
        # Generate bash script for this chunk
        bash_script = generate_bash_script(chunk, args.robot_name)
        
        # Read template and modify it
        output_lines = []
        in_args_section = False
        skip_lines = 0
        
        for idx, line in enumerate(template_lines):
            if skip_lines > 0:
                skip_lines -= 1
                continue
                
            # Update generateName
            if 'generateName:' in line:
                job_suffix = f"-{args.robot_name.replace('_', '-')}-{i+1}-"[:40]
                output_lines.append(f"  generateName: groot-finetune{job_suffix}\n")
                continue
            
            # Update WANDB_PROJECT
            if 'name: WANDB_PROJECT' in line:
                output_lines.append(line)
                # Next line should be value
                if idx + 1 < len(template_lines) and 'value:' in template_lines[idx + 1]:
                    output_lines.append(f"          value: {args.robot_name}\n")
                    skip_lines = 1
                continue
            
            # Replace args section
            if 'args:' in line and 'command:' not in line:
                output_lines.append(line)
                # Next line should be "- |"
                if idx + 1 < len(template_lines) and template_lines[idx + 1].strip() == '- |':
                    output_lines.append(template_lines[idx + 1])
                    skip_lines = 1
                    # Add the generated bash script with proper indentation
                    for bash_line in bash_script.split('\n'):
                        output_lines.append(f"            {bash_line}\n")
                    # Skip until we find the next major section (restartPolicy)
                    j = idx + 2
                    while j < len(template_lines) and 'restartPolicy:' not in template_lines[j]:
                        j += 1
                        skip_lines += 1
                continue
            
            output_lines.append(line)

        # Save to file
        output_filename = f"generated_jobs/masstune_{args.robot_name}_{timestamp}_{i+1}.yaml"
        with open(output_filename, 'w') as f:
            f.writelines(output_lines)
        
        generated_files.append(output_filename)

    # 5. Output and Run Commands
    print("\nGenerated Job Files:")
    for gf in generated_files:
        print(f" - {gf}")

    print("\nTo submit jobs, run:")
    for gf in generated_files:
        print(f"  oc create -f {gf}")

if __name__ == "__main__":
    main()
