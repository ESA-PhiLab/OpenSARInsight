#!/bin/bash
# Copyright 2025 [Your Name/Organization]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Project setup and environment configuration script
# This script sets up the development environment for the project

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.9"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if ! command_exists python3; then
        error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Check Python version
    local python_ver
    python_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_ver >= $PYTHON_VERSION" | bc -l) -eq 0 ]]; then
        error "Python $PYTHON_VERSION+ is required. Found: $python_ver"
        exit 1
    fi
    
    success "Python $python_ver found"
    
    # Check pip
    if ! command_exists pip3; then
        warning "pip3 not found. Installing..."
        python3 -m ensurepip --upgrade
    fi
    
    success "System requirements check completed"
}

# Create virtual environment
create_venv() {
    log "Setting up virtual environment..."
    
    local venv_path="$PROJECT_ROOT/venv"
    
    if [[ -d "$venv_path" ]]; then
        warning "Virtual environment already exists at $venv_path"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$venv_path"
        else
            log "Using existing virtual environment"
            return 0
        fi
    fi
    
    python3 -m venv "$venv_path"
    success "Virtual environment created at $venv_path"
}

# Install dependencies
install_dependencies() {
    log "Installing project dependencies..."
    
    local venv_path="$PROJECT_ROOT/venv"
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install project in development mode
    pip install -e "[$PROJECT_ROOT[dev]]"
    
    success "Dependencies installed successfully"
}

# Setup git hooks (optional)
setup_git_hooks() {
    log "Setting up git hooks..."
    
    if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
        warning "Not a git repository. Skipping git hooks setup."
        return 0
    fi
    
    # Create pre-commit hook for linting
    local hook_file="$PROJECT_ROOT/.git/hooks/pre-commit"
    
    cat > "$hook_file" << 'EOF'
#!/bin/bash
# Pre-commit hook for code quality checks

set -e

echo "Running pre-commit checks..."

# Check Python files with basic linting
if command -v python3 >/dev/null 2>&1; then
    python3 -m py_compile $(find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*")
    echo "✓ Python syntax check passed"
fi

echo "Pre-commit checks completed successfully"
EOF
    
    chmod +x "$hook_file"
    success "Git hooks setup completed"
}

# Create necessary directories
create_directories() {
    log "Creating project directories..."
    
    local dirs=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/tmp"
        "$PROJECT_ROOT/output"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
    
    success "Directory structure setup completed"
}

# Main setup function
main() {
    log "Starting project setup..."
    log "Project root: $PROJECT_ROOT"
    
    check_requirements
    create_venv
    install_dependencies
    setup_git_hooks
    create_directories
    
    success "Project setup completed successfully!"
    log ""
    log "To activate the virtual environment, run:"
    log "  source venv/bin/activate"
    log ""
    log "To run tests:"
    log "  python -m pytest tests/"
    log ""
    log "To run a Python script:"
    log "  python pyscripts/script_name.py"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -h, --help     Show this help message"
            echo "  --no-hooks     Skip git hooks setup"
            echo ""
            echo "This script sets up the development environment for the project."
            exit 0
            ;;
        --no-hooks)
            SKIP_HOOKS=1
            shift
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
