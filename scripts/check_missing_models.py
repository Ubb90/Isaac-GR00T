#!/usr/bin/env python3
import os
import argparse
import re
from pathlib import Path

def get_output_name(run_name, ckpt_name):
    """
    Derives output name from run name and checkpoint name.
    Matches logic in full_recorder.py (ignoring data_config for now as it's not in the folder structure)
    """
    match = re.search(r'checkpoint-(\d+)', ckpt_name)
    if match:
        base_name = f"{run_name}_{match.group(1)}"
    else:
        base_name = f"{run_name}_{ckpt_name}"
    return base_name

def main():
    parser = argparse.ArgumentParser(description="Find models in source that are not in destination")
    parser.add_argument("source_dir", help="Directory containing run folders with checkpoints (e.g. /media/baxter/storage/models/groot)")
    parser.add_argument("dest_dir", help="Directory containing processed output folders (e.g. /media/baxter/T7RawData/tmp1)")
    parser.add_argument("--output", default="missing_models.txt", help="Output file for missing models list")
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)
    
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist")
        return

    # Create dest path if it doesn't exist (implies all are missing)
    if not dest_path.exists():
        print(f"Warning: Destination path {dest_path} does not exist. All models will be considered missing.")

    missing_models = []

    print(f"Scanning {source_path} for checkpoints...")
    
    # Walk source directory
    # Structure expected: source_dir/run_name/checkpoint-N
    for run_dir in source_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        run_name = run_dir.name
        
        # Look for checkpoints inside run_dir
        for ckpt_dir in run_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue
                
            if "checkpoint-" in ckpt_dir.name:
                # Check if config.json exists in the checkpoint folder
                if not (ckpt_dir / "config.json").exists():
                    continue

                # This is a checkpoint
                output_name = get_output_name(run_name, ckpt_dir.name)
                output_path = dest_path / output_name
                
                if not output_path.exists():
                    # It's missing
                    missing_models.append(str(ckpt_dir.absolute()))

    # Write to file
    output_file = Path(os.getcwd()) / args.output
    with open(output_file, 'w') as f:
        for model_path in sorted(missing_models):
            f.write(f"{model_path}\n")
            
    print(f"Found {len(missing_models)} missing models.")
    print(f"List written to {output_file}")

if __name__ == "__main__":
    main()
