#!/bin/bash
# Build script for the full_recorder Docker container with checkpoint debugging support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

IMAGE_NAME="groot-recorder"
DOCKERFILE="Dockerfile.full_recorder"

print_usage() {
    echo -e "${BLUE}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo "Options:"
    echo "  --full              Build complete image (default)"
    echo "  --test              Build and verify all checkpoints"
    echo "  --interactive       Build and run interactive shell"
    echo "  --run               Build and run the full_recorder script"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --full                 # Build complete image"
    echo "  $0 --test                 # Test all checkpoints"
    echo "  $0 --interactive          # Build and open shell"
    echo "  $0 --run                  # Build and run recorder script"
}

echo_step() {
    echo -e "${BLUE}==>${NC} $1"
}

echo_success() {
    echo -e "${GREEN}✓${NC} $1"
}

echo_error() {
    echo -e "${RED}✗${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

test_checkpoints() {
    echo_step "Testing Docker build at each checkpoint..."
    
    # Since we're not using multi-stage builds, we'll build the full image
    # and verify by checking the logs
    
    echo_step "Building full image and verifying checkpoints..."
    if docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:test" . 2>&1 | tee /tmp/docker_build.log; then
        echo ""
        echo_step "Verifying checkpoints in build log..."
        
        for i in {1..9}; do
            if grep -q "CHECKPOINT $i:.*✓" /tmp/docker_build.log; then
                echo_success "Checkpoint $i passed"
            else
                echo_error "Checkpoint $i failed or not found"
                return 1
            fi
        done
        
        echo ""
        echo_success "All checkpoints verified successfully!"
        return 0
    else
        echo_error "Build failed!"
        return 1
    fi
}

build_full() {
    echo_step "Building full Docker image..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:latest" .
    echo_success "Build complete: $IMAGE_NAME:latest"
    
    echo ""
    echo_step "Image details:"
    docker images "$IMAGE_NAME:latest"
    
    echo ""
    echo_step "Verifying conda environments in image..."
    docker run --rm "$IMAGE_NAME:latest" /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda env list"
}

run_interactive() {
    echo_step "Building and launching interactive shell..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:latest" .
    
    echo ""
    echo_success "Build complete. Starting interactive shell..."
    echo_warning "Note: You'll need to mount volumes for full functionality"
    echo ""
    
    docker run --gpus all -it --rm \
        -v "$(pwd)":/workspace/Isaac-GR00T \
        "$IMAGE_NAME:latest" /bin/bash
}

run_recorder() {
    echo_step "Building and running full_recorder script..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:latest" .
    
    echo ""
    echo_success "Build complete. Starting recorder..."
    echo_warning "Make sure to mount required volumes!"
    echo ""
    
    # Example run with volume mounts - adjust paths as needed
    docker run --gpus all --rm \
        -v "$(pwd)":/workspace/Isaac-GR00T \
        -v "$HOME/Documents/LeTrack":/workspace/LeTrack \
        -v "/media/baxter/T7RawData":/media/baxter/T7RawData \
        -v "/media/baxter/storage":/media/baxter/storage \
        "$IMAGE_NAME:latest"
}

# Main script logic
MODE="full"

if [ $# -eq 0 ]; then
    build_full
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --full)
            build_full
            exit 0
            ;;
        --test)
            test_checkpoints
            exit $?
            ;;
        --interactive)
            run_interactive
            exit 0
            ;;
        --run)
            run_recorder
            exit 0
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
    shift
done
