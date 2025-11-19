#!/bin/bash

# Source Code Snapshot Generator for Isaac GR00T
# Creates a comprehensive markdown file containing all pertinent source code

OUTPUT_FILE="source_snapshot.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Directories to include
INCLUDE_DIRS=(
    "gr00t"
    "scripts"
    "tests"
    "examples"
    "deployment_scripts"
)

# File extensions to include
INCLUDE_EXTENSIONS=(
    "*.py"
    "*.yaml"
    "*.yml"
    "*.json"
    "*.toml"
    "*.txt"
    "*.sh"
    "*.md"
    "*.cfg"
    "*.ini"
)

# Directories to exclude (relative patterns)
EXCLUDE_DIRS=(
    "__pycache__"
    ".git"
    ".pytest_cache"
    "*.egg-info"
    "build"
    "dist"
    ".venv"
    "venv"
    "env"
    "node_modules"
    ".ipynb_checkpoints"
    "demo_data"
    "data"
    "datasets"
    "checkpoints"
    "outputs"
    "logs"
    "wandb"
    ".ruff_cache"
    ".mypy_cache"
)

# Files to exclude
EXCLUDE_FILES=(
    "*.pyc"
    "*.pyo"
    "*.pyd"
    "*.so"
    "*.pkl"
    "*.pt"
    "*.pth"
    "*.bin"
    "*.safetensors"
    "*.ckpt"
    "*.h5"
    "*.hdf5"
    "*.parquet"
    "*.arrow"
    "*.mp4"
    "*.avi"
    "*.jpg"
    "*.jpeg"
    "*.png"
    "*.gif"
    "*.pdf"
    "*.zip"
    "*.tar"
    "*.gz"
)

# Important root-level files to always include
ROOT_FILES=(
    "README.md"
    "CLAUDE.md"
    "setup.py"
    "pyproject.toml"
    "requirements.txt"
    "Makefile"
    ".gitignore"
    "LICENSE"
)

# Function to check if file should be excluded
should_exclude_file() {
    local file="$1"
    local basename=$(basename "$file")

    # Check file patterns
    for pattern in "${EXCLUDE_FILES[@]}"; do
        if [[ "$basename" == $pattern ]]; then
            return 0
        fi
    done

    # Check if file is too large (> 1MB)
    if [ -f "$file" ]; then
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt 1048576 ]; then
            return 0
        fi
    fi

    return 1
}

# Function to check if directory should be excluded
should_exclude_dir() {
    local dir="$1"
    local basename=$(basename "$dir")

    for pattern in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$basename" == $pattern ]] || [[ "$dir" == *"/$pattern"* ]]; then
            return 0
        fi
    done

    return 1
}

# Function to get file language for syntax highlighting
get_language() {
    local file="$1"
    case "${file##*.}" in
        py) echo "python" ;;
        sh) echo "bash" ;;
        yaml|yml) echo "yaml" ;;
        json) echo "json" ;;
        toml) echo "toml" ;;
        md) echo "markdown" ;;
        txt) echo "text" ;;
        cfg|ini) echo "ini" ;;
        *) echo "" ;;
    esac
}

# Function to add file to snapshot
add_file_to_snapshot() {
    local file="$1"
    local rel_path="${file#./}"
    local language=$(get_language "$file")

    echo "" >> "$OUTPUT_FILE"
    echo "## \`$rel_path\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "\`\`\`$language" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
}

# Initialize output file
echo "# Isaac GR00T Source Code Snapshot" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Generated:** $TIMESTAMP" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "This document contains a comprehensive snapshot of the source code in the Isaac GR00T repository." >> "$OUTPUT_FILE"
echo "It is intended to provide context to LLM agents for understanding and working with this codebase." >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add table of contents placeholder
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "# Table of Contents" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Collect all files first for TOC
echo "Scanning repository..." >&2
all_files=()

# Add root-level important files
for file in "${ROOT_FILES[@]}"; do
    if [ -f "$file" ]; then
        all_files+=("$file")
    fi
done

# Scan include directories
for dir in "${INCLUDE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Scanning $dir/..." >&2
        while IFS= read -r -d '' file; do
            if ! should_exclude_file "$file"; then
                all_files+=("$file")
            fi
        done < <(find "$dir" -type f \( $(printf " -name %s -o" "${INCLUDE_EXTENSIONS[@]}" | sed 's/ -o$//') \) -print0 2>/dev/null)
    fi
done

# Remove files in excluded directories
filtered_files=()
for file in "${all_files[@]}"; do
    exclude=0
    for excl_dir in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$file" == *"/$excl_dir/"* ]] || [[ "$file" == *"/$excl_dir" ]]; then
            exclude=1
            break
        fi
    done
    if [ $exclude -eq 0 ]; then
        filtered_files+=("$file")
    fi
done

# Sort files
IFS=$'\n' sorted_files=($(sort <<<"${filtered_files[*]}"))
unset IFS

# Generate TOC
echo "Generating table of contents..." >&2
for file in "${sorted_files[@]}"; do
    rel_path="${file#./}"
    # Convert to markdown anchor
    anchor=$(echo "$rel_path" | tr '[:upper:]' '[:lower:]' | tr '/' '-' | tr '.' '-' | tr '_' '-')
    echo "- [\`$rel_path\`](#$anchor)" >> "$OUTPUT_FILE"
done

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "# Source Files" >> "$OUTPUT_FILE"

# Add all files to snapshot
total=${#sorted_files[@]}
current=0
for file in "${sorted_files[@]}"; do
    current=$((current + 1))
    echo "Processing ($current/$total): $file" >&2
    add_file_to_snapshot "$file"
done

# Add summary at the end
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "# Summary" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Total files included:** $total" >> "$OUTPUT_FILE"
echo "**Generated:** $TIMESTAMP" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "This snapshot includes:" >> "$OUTPUT_FILE"
for dir in "${INCLUDE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "- \`$dir/\` directory" >> "$OUTPUT_FILE"
    fi
done
echo "- Root-level configuration and documentation files" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Excluded: binaries, model weights, datasets, cache files, and generated artifacts." >> "$OUTPUT_FILE"

echo "" >&2
echo "✓ Source snapshot created: $OUTPUT_FILE" >&2
echo "  Total files: $total" >&2
echo "  File size: $(du -h "$OUTPUT_FILE" | cut -f1)" >&2
