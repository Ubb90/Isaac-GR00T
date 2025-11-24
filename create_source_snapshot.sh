#!/bin/bash

# Source Code Snapshot Generator for Isaac GR00T
# Creates a comprehensive markdown file containing all pertinent source code
# Optimized for LLM coding agent context

OUTPUT_FILE="source_snapshot.md"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Directories to include
INCLUDE_DIRS=(
    "gr00t"
    "scripts"
    "tests"
    "examples"
    "deployment_scripts"
    "getting_started"
    "ros_ws"
    ".github"
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
    "*.xml"
    "*.launch"
    "*.urdf"
    "*.xacro"
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
    "media"
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
    "source_snapshot.md"
    "Dockerfile.bak"
)

# Important root-level files to always include
ROOT_FILES=(
    "README.md"
    "CLAUDE.md"
    "CONTRIBUTING.md"
    "LICENSE"
    "setup.py"
    "pyproject.toml"
    "requirements.txt"
    "Makefile"
    ".gitignore"
    ".gitattributes"
    "Dockerfile"
    "orin.Dockerfile"
    "thor.Dockerfile"
    "docker-compose.yml"
    "docker-compose.yaml"
    "groot-finetune-job.yaml"
    ".dockerignore"
    ".env.example"
    "setup.cfg"
    "tox.ini"
    "poetry.lock"
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
        xml|launch|urdf|xacro) echo "xml" ;;
        Dockerfile) echo "dockerfile" ;;
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

# Add environment context
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "# Environment & Context" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# System information
echo "## System Information" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "Timestamp: $TIMESTAMP" >> "$OUTPUT_FILE"
echo "OS: $(uname -s)" >> "$OUTPUT_FILE"
echo "Kernel: $(uname -r)" >> "$OUTPUT_FILE"
echo "Architecture: $(uname -m)" >> "$OUTPUT_FILE"
if command -v python3 &> /dev/null; then
    echo "Python: $(python3 --version 2>&1)" >> "$OUTPUT_FILE"
fi
if command -v python &> /dev/null; then
    echo "Python (fallback): $(python --version 2>&1)" >> "$OUTPUT_FILE"
fi
if command -v pip3 &> /dev/null; then
    echo "pip: $(pip3 --version 2>&1)" >> "$OUTPUT_FILE"
fi
if command -v poetry &> /dev/null; then
    echo "Poetry: $(poetry --version 2>&1)" >> "$OUTPUT_FILE"
fi
if command -v docker &> /dev/null; then
    echo "Docker: $(docker --version 2>&1)" >> "$OUTPUT_FILE"
fi
if command -v git &> /dev/null; then
    echo "Git: $(git --version 2>&1)" >> "$OUTPUT_FILE"
fi
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Git repository information
if [ -d ".git" ]; then
    echo "## Git Repository Information" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "Repository: $(git config --get remote.origin.url 2>/dev/null || echo 'N/A')" >> "$OUTPUT_FILE"
    echo "Current Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')" >> "$OUTPUT_FILE"
    echo "Latest Commit: $(git log -1 --format='%H' 2>/dev/null || echo 'N/A')" >> "$OUTPUT_FILE"
    echo "Commit Message: $(git log -1 --format='%s' 2>/dev/null || echo 'N/A')" >> "$OUTPUT_FILE"
    echo "Author: $(git log -1 --format='%an <%ae>' 2>/dev/null || echo 'N/A')" >> "$OUTPUT_FILE"
    echo "Date: $(git log -1 --format='%ad' 2>/dev/null || echo 'N/A')" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "Recent Commits (last 10):" >> "$OUTPUT_FILE"
    git log -10 --oneline 2>/dev/null >> "$OUTPUT_FILE" || echo "N/A" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "Git Status:" >> "$OUTPUT_FILE"
    git status --short 2>/dev/null >> "$OUTPUT_FILE" || echo "N/A" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Python dependencies
echo "## Python Dependencies" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ -f "requirements.txt" ]; then
    echo "### requirements.txt" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    cat requirements.txt >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

if [ -f "pyproject.toml" ]; then
    echo "### pyproject.toml (dependencies section)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```toml' >> "$OUTPUT_FILE"
    cat pyproject.toml >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Installed packages (if pip is available)
if command -v pip3 &> /dev/null; then
    echo "### Installed Packages (pip freeze)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    pip3 freeze 2>/dev/null >> "$OUTPUT_FILE" || echo "Unable to retrieve installed packages" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Poetry lock info (if available)
if [ -f "poetry.lock" ] && command -v poetry &> /dev/null; then
    echo "### Poetry Dependencies" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    poetry show 2>/dev/null >> "$OUTPUT_FILE" || echo "Unable to retrieve Poetry dependencies" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Project structure
echo "## Project Directory Structure" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
if command -v tree &> /dev/null; then
    tree -L 3 -I '__pycache__|*.pyc|.git|.pytest_cache|*.egg-info|build|dist|venv|env|node_modules|demo_data|data|datasets|checkpoints|outputs|logs|wandb|.ruff_cache|.mypy_cache|media' 2>/dev/null >> "$OUTPUT_FILE" || echo "Tree command failed" >> "$OUTPUT_FILE"
else
    # Fallback to find-based tree
    find . -maxdepth 3 -not -path '*/\.*' -not -path '*/__pycache__/*' -not -path '*/venv/*' -not -path '*/env/*' -not -path '*/demo_data/*' -not -path '*/data/*' -not -path '*/datasets/*' -not -path '*/checkpoints/*' -not -path '*/outputs/*' -not -path '*/logs/*' -not -path '*/wandb/*' -not -path '*/media/*' 2>/dev/null | sort >> "$OUTPUT_FILE" || echo "Unable to generate directory structure" >> "$OUTPUT_FILE"
fi
echo '```' >> "$OUTPUT_FILE"
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

        # Build find command with proper quoting
        find_args=()
        for ext in "${INCLUDE_EXTENSIONS[@]}"; do
            if [ ${#find_args[@]} -gt 0 ]; then
                find_args+=("-o")
            fi
            find_args+=("-name" "$ext")
        done

        while IFS= read -r -d '' file; do
            if ! should_exclude_file "$file"; then
                all_files+=("$file")
            fi
        done < <(find "$dir" -type f \( "${find_args[@]}" \) -print0 2>/dev/null)
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

# Add summary and statistics at the end
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "# Summary & Statistics" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "## File Statistics" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Total files included:** $total" >> "$OUTPUT_FILE"
echo "**Generated:** $TIMESTAMP" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Count files by type
echo "### Files by Extension" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
for file in "${sorted_files[@]}"; do
    echo "${file##*.}"
done | sort | uniq -c | sort -rn >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Count files by directory
echo "### Files by Directory" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
for file in "${sorted_files[@]}"; do
    dirname "$file"
done | sort | uniq -c | sort -rn >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Code statistics (lines of code)
if command -v wc &> /dev/null; then
    echo "### Lines of Code" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"

    total_lines=0
    python_lines=0
    yaml_lines=0
    sh_lines=0
    md_lines=0

    for file in "${sorted_files[@]}"; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file" 2>/dev/null || echo 0)
            total_lines=$((total_lines + lines))

            case "${file##*.}" in
                py) python_lines=$((python_lines + lines)) ;;
                yaml|yml) yaml_lines=$((yaml_lines + lines)) ;;
                sh) sh_lines=$((sh_lines + lines)) ;;
                md) md_lines=$((md_lines + lines)) ;;
            esac
        fi
    done

    echo "Total lines: $total_lines" >> "$OUTPUT_FILE"
    echo "Python (.py): $python_lines" >> "$OUTPUT_FILE"
    echo "YAML (.yaml/.yml): $yaml_lines" >> "$OUTPUT_FILE"
    echo "Shell (.sh): $sh_lines" >> "$OUTPUT_FILE"
    echo "Markdown (.md): $md_lines" >> "$OUTPUT_FILE"
    echo "Other: $((total_lines - python_lines - yaml_lines - sh_lines - md_lines))" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

echo "## Included Content" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "This snapshot includes:" >> "$OUTPUT_FILE"
for dir in "${INCLUDE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f 2>/dev/null | wc -l)
        echo "- \`$dir/\` directory ($file_count files)" >> "$OUTPUT_FILE"
    fi
done
echo "- Root-level configuration and documentation files" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "## Excluded Content" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Excluded from snapshot:" >> "$OUTPUT_FILE"
echo "- Binaries and compiled files (.pyc, .so, .bin, etc.)" >> "$OUTPUT_FILE"
echo "- Model weights and checkpoints (.pt, .pth, .ckpt, .safetensors, etc.)" >> "$OUTPUT_FILE"
echo "- Data files and datasets (datasets/, data/, demo_data/)" >> "$OUTPUT_FILE"
echo "- Media files (.jpg, .png, .gif, .mp4, media/)" >> "$OUTPUT_FILE"
echo "- Cache and temporary files (__pycache__, .pytest_cache, etc.)" >> "$OUTPUT_FILE"
echo "- Build artifacts (build/, dist/, *.egg-info/)" >> "$OUTPUT_FILE"
echo "- Virtual environments (venv/, env/, .venv/)" >> "$OUTPUT_FILE"
echo "- Logs and outputs (logs/, outputs/, wandb/)" >> "$OUTPUT_FILE"
echo "- Version control (.git/)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "## Snapshot File Information" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "Output file: $OUTPUT_FILE" >> "$OUTPUT_FILE"
if [ -f "$OUTPUT_FILE" ]; then
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)" >> "$OUTPUT_FILE"
    echo "Line count: $(wc -l < "$OUTPUT_FILE")" >> "$OUTPUT_FILE"
fi
echo '```' >> "$OUTPUT_FILE"

echo "" >&2
echo "✓ Source snapshot created: $OUTPUT_FILE" >&2
echo "  Total files: $total" >&2
echo "  File size: $(du -h "$OUTPUT_FILE" | cut -f1)" >&2
echo "  Total lines in snapshot: $(wc -l < "$OUTPUT_FILE")" >&2
