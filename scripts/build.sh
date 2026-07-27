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

# Build script for the project
# This script handles building, testing, and packaging of the project

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"

# Default options
RUN_TESTS=true
RUN_LINT=true
CREATE_PACKAGE=false
CLEAN_BUILD=false
VERBOSE=false

# Logging functions
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

verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${NC}[VERBOSE]${NC} $1"
    fi
}

# Check if virtual environment is activated
check_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        warning "Virtual environment not detected."
        if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
            log "Activating virtual environment..."
            # shellcheck source=/dev/null
            source "$PROJECT_ROOT/venv/bin/activate"
        else
            error "No virtual environment found. Run ./scripts/setup.sh first."
            exit 1
        fi
    fi
    verbose "Using virtual environment: $VIRTUAL_ENV"
}

# Clean build directories
clean_build() {
    log "Cleaning build directories..."
    
    local dirs_to_clean=(
        "$BUILD_DIR"
        "$DIST_DIR"
        "$PROJECT_ROOT/*.egg-info"
        "$PROJECT_ROOT/__pycache__"
        "$PROJECT_ROOT/**/__pycache__"
        "$PROJECT_ROOT/.pytest_cache"
        "$PROJECT_ROOT/.coverage"
        "$PROJECT_ROOT/htmlcov"
    )
    
    for pattern in "${dirs_to_clean[@]}"; do
        if ls $pattern 1> /dev/null 2>&1; then
            rm -rf $pattern
            verbose "Removed: $pattern"
        fi
    done
    
    success "Build directories cleaned"
}

# Run code linting
run_lint() {
    if [[ "$RUN_LINT" != true ]]; then
        return 0
    fi
    
    log "Running code linting..."
    
    # Check if flake8 is available
    if command -v flake8 >/dev/null 2>&1; then
        flake8 src/ pyscripts/ tests/ --max-line-length=100 --extend-ignore=E203,W503
        success "Linting passed"
    else
        warning "flake8 not installed. Skipping linting."
    fi
}

# Run tests
run_tests() {
    if [[ "$RUN_TESTS" != true ]]; then
        return 0
    fi
    
    log "Running tests..."
    
    # Check if pytest is available
    if command -v pytest >/dev/null 2>&1; then
        local pytest_args=()
        
        if [[ "$VERBOSE" == true ]]; then
            pytest_args+=("-v")
        fi
        
        # Run tests with coverage if available
        if python -c "import pytest_cov" 2>/dev/null; then
            pytest_args+=("--cov=src" "--cov-report=html" "--cov-report=term")
        fi
        
        pytest "${pytest_args[@]}" tests/
        success "All tests passed"
    else
        warning "pytest not installed. Skipping tests."
    fi
}

# Build package
build_package() {
    if [[ "$CREATE_PACKAGE" != true ]]; then
        return 0
    fi
    
    log "Building package..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    
    # Build wheel and source distribution
    python -m build --outdir "$DIST_DIR"
    
    success "Package built successfully"
    log "Package files:"
    ls -la "$DIST_DIR"
}

# Install package in development mode
install_dev() {
    log "Installing package in development mode..."
    pip install -e ".[dev]"
    success "Package installed in development mode"
}

# Generate documentation
generate_docs() {
    log "Generating documentation..."
    
    if [[ -f "$PROJECT_ROOT/docs/requirements.txt" ]]; then
        pip install -r "$PROJECT_ROOT/docs/requirements.txt"
    fi
    
    # Check if Sphinx is available
    if command -v sphinx-build >/dev/null 2>&1; then
        local docs_source="$PROJECT_ROOT/docs"
        local docs_build="$PROJECT_ROOT/docs/_build"
        
        sphinx-build -b html "$docs_source" "$docs_build"
        success "Documentation generated at $docs_build/index.html"
    else
        warning "Sphinx not installed. Skipping documentation generation."
    fi
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
    -h, --help              Show this help message
    -c, --clean             Clean build directories before building
    -p, --package           Create distribution package
    --no-tests              Skip running tests
    --no-lint               Skip code linting
    -v, --verbose           Enable verbose output
    --docs                  Generate documentation
    --install-dev           Install package in development mode

Examples:
    $0                      # Run tests and linting
    $0 --clean --package    # Clean, test, lint, and create package
    $0 --no-tests --docs    # Skip tests, generate documentation
EOF
}

# Main build function
main() {
    log "Starting build process..."
    log "Project root: $PROJECT_ROOT"
    
    check_venv
    
    if [[ "$CLEAN_BUILD" == true ]]; then
        clean_build
    fi
    
    run_lint
    run_tests
    
    if [[ "$GENERATE_DOCS" == true ]]; then
        generate_docs
    fi
    
    if [[ "$INSTALL_DEV" == true ]]; then
        install_dev
    fi
    
    build_package
    
    success "Build process completed successfully!"
}

# Parse command line arguments
GENERATE_DOCS=false
INSTALL_DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -p|--package)
            CREATE_PACKAGE=true
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --no-lint)
            RUN_LINT=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --docs)
            GENERATE_DOCS=true
            shift
            ;;
        --install-dev)
            INSTALL_DEV=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run main function
main
