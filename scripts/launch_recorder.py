import argparse
import os
import math
import yaml
import subprocess
import copy
import glob
from datetime import datetime

def generate_bash_script(models_chunk):
    # Join the models with newlines
    models_str = "\n".join(models_chunk)
    
    script = f"""
echo "--- SETUP ---"
source /opt/conda/etc/profile.d/conda.sh
conda activate gr00t

echo "--- WRITING MODEL LIST ---"
cat <<EOF > missing_models.txt
{models_str}
EOF

echo "--- STARTING RECORDER ---"
if [ -s missing_models.txt ]; then
    cat missing_models.txt
    python scripts/full_recorder.py --config-list missing_models.txt
else
    echo "No models in list. Exiting."
fi
"""
    return script

def main():
    parser = argparse.ArgumentParser(description="Generate and run Groot recorder jobs from a list of models.")
    parser.add_argument("--models-file", type=str, default="missing_models.txt", help="Path to the missing models text file.")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs/Jobs to split the workload across")
    parser.add_argument("--template", type=str, default="groot-recorder.yaml", help="Path to the template YAML file")
    
    args = parser.parse_args()

    # 1. Read models list
    if not os.path.exists(args.models_file):
        print(f"Models file {args.models_file} not found.")
        return

    with open(args.models_file, 'r') as f:
        # Filter out empty lines using strip()
        all_models = [line.strip() for line in f if line.strip()]

    # Transform paths from local to cluster
    # Local: /media/baxter/storage/models/groot
    # Cluster: /mnt/primary/groot
    local_prefix = '/media/baxter/storage/models/groot'
    cluster_prefix = '/media/baxter/storage/models/groot'
    all_models = [m.replace(local_prefix, cluster_prefix) for m in all_models]

    if not all_models:
        print(f"No models found in {args.models_file}")
        return

    print(f"Found {len(all_models)} models to record.")

    # 2. Split workload
    # Ensure we don't create more jobs than there are models
    num_jobs = min(args.num_gpus, len(all_models))
    if num_jobs < 1:
        num_jobs = 1
        
    chunk_size = math.ceil(len(all_models) / num_jobs)
    model_chunks = [all_models[i:i + chunk_size] for i in range(0, len(all_models), chunk_size)]

    print(f"Splitting into {len(model_chunks)} jobs.")

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
    if os.path.exists("generated_jobs"):
        for f in glob.glob("generated_jobs/*"):
            try:
                os.remove(f)
            except OSError:
                pass
    os.makedirs("generated_jobs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for i, chunk in enumerate(model_chunks):
        # Save split text file
        split_model_filename = f"generated_jobs/models_chunk_{timestamp}_{i+1}.txt"
        with open(split_model_filename, 'w') as f:
            f.write("\n".join(chunk))

        job_yaml = copy.deepcopy(template_content)
        
        # Update metadata to ensure unique names or rely on generateName
        # Default groot-recorder.yaml has generateName: groot-recorder-
        # We can append a suffix or let k8s handle it, but it's good to distinguish them if needed.
        # But generateName takes care of uniqueness if we submit multiple.
        # However, to be safe and clear, let's keep generateName but maybe update it slightly?
        # Actually, if we use the same generateName, they will just be groot-recorder-xxxxx. 
        # That's fine.
        
        # Update command/args
        containers = job_yaml['spec']['template']['spec']['containers']
        # Assume the first container is the one we want
        container = containers[0]
        
        # We replace the args with our generated script
        bash_script = generate_bash_script(chunk)
        
        # groot-recorder.yaml uses command: ["/bin/bash", "-c"] and args: [script]
        # We preserve that structure
        container['args'] = [bash_script]

        # Save to file
        output_filename = f"generated_jobs/recorder_job_{timestamp}_{i+1}.yaml"
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
        inp = input(f"Executing: {cmd}")
        if inp.lower() in ['y', 'yes', '']:
            print(f"Executing: {cmd}")
        else:
            print("Skipping execution.")
            continue
        # Execute the command
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
