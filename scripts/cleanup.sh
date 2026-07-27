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

# Cleanup script for the project
# This script removes build artifacts, cache files, and temporary data

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

# Default options
CLEAN_ALL=false
CLEAN_BUILD=false
CLEAN_CACHE=false
CLEAN_DATA=false
CLEAN_LOGS=false
DRY_RUN=false
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

# Remove files/directories safely
safe_remove() {
    local path="$1"
    local description="${2:-item}"
    
    if [[ -e "$path" ]]; then
        local size=""
        if [[ -f "$path" ]]; then
            size=" ($(du -h "$path" | cut -f1))"
        elif [[ -d "$path" ]]; then
            size=" ($(du -sh "$path" | cut -f1))"
        fi
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "Would remove $description: $path$size"
        else
            verbose "Removing $description: $path$size"
            rm -rf "$path"
            success "Removed $description: $path"
        fi
    else
        verbose "$description not found: $path"
    fi
}

# Clean build artifacts
clean_build_artifacts() {
    if [[ "$CLEAN_BUILD" != true && "$CLEAN_ALL" != true ]]; then
        return 0
    fi
    
    log "Cleaning build artifacts..."
    
    local build_patterns=(
        "$PROJECT_ROOT/build"
        "$PROJECT_ROOT/dist"
        "$PROJECT_ROOT/*.egg-info"
        "$PROJECT_ROOT/src/*.egg-info"
        "$PROJECT_ROOT/.eggs"
    )
    
    for pattern in "${build_patterns[@]}"; do
        for path in $pattern; do
            if [[ -e "$path" ]]; then
                safe_remove "$path" "build artifact"
            fi
        done
    done
    
    success "Build artifacts cleanup completed"
}

# Clean cache files
clean_cache_files() {
    if [[ "$CLEAN_CACHE" != true && "$CLEAN_ALL" != true ]]; then
        return 0
    fi
    
    log "Cleaning cache files..."
    
    # Python cache files
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyo" -delete 2>/dev/null || true
    
    # Test and coverage cache
    local cache_patterns=(
        "$PROJECT_ROOT/.pytest_cache"
        "$PROJECT_ROOT/.coverage"
        "$PROJECT_ROOT/htmlcov"
        "$PROJECT_ROOT/.tox"
        "$PROJECT_ROOT/.mypy_cache"
        "$PROJECT_ROOT/.ruff_cache"
    )
    
    for pattern in "${cache_patterns[@]}"; do
        safe_remove "$pattern" "cache directory"
    done
    
    # Jupyter notebook checkpoints
    find "$PROJECT_ROOT" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    
    # macOS specific files
    find "$PROJECT_ROOT" -name ".DS_Store" -delete 2>/dev/null || true
    
    success "Cache files cleanup completed"
}

# Clean temporary data
clean_temp_data() {
    if [[ "$CLEAN_DATA" != true && "$CLEAN_ALL" != true ]]; then
        return 0
    fi
    
    log "Cleaning temporary data..."
    
    local temp_dirs=(
        "$PROJECT_ROOT/tmp"
        "$PROJECT_ROOT/temp"
        "$PROJECT_ROOT/data/tmp"
        "$PROJECT_ROOT/data/temp"
        "$PROJECT_ROOT/output/tmp"
    )
    
    for dir in "${temp_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "Would clean contents of: $dir"
                ls -la "$dir" 2>/dev/null || true
            else
                find "$dir" -mindepth 1 -delete 2>/dev/null || true
                verbose "Cleaned contents of: $dir"
            fi
        fi
    done
    
    success "Temporary data cleanup completed"
}

# Clean log files
clean_log_files() {
    if [[ "$CLEAN_LOGS" != true && "$CLEAN_ALL" != true ]]; then
        return 0
    fi
    
    log "Cleaning log files..."
    
    local log_patterns=(
        "$PROJECT_ROOT/logs/*.log"
        "$PROJECT_ROOT/*.log"
        "$PROJECT_ROOT/output/*.log"
    )
    
    for pattern in "${log_patterns[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]]; then
                safe_remove "$file" "log file"
            fi
        done
    done
    
    success "Log files cleanup completed"
}

# Show what would be cleaned
show_dry_run_summary() {
    if [[ "$DRY_RUN" != true ]]; then
        return 0
    fi
    
    log ""
    log "Dry run completed. Use without --dry-run to actually remove files."
    log "Summary of items that would be cleaned:"
    
    local total_size=0
    
    # Calculate potential savings (simplified)
    if command -v du >/dev/null 2>&1; then
        local patterns=(
            "$PROJECT_ROOT/build"
            "$PROJECT_ROOT/dist"
            "$PROJECT_ROOT/.pytest_cache"
            "$PROJECT_ROOT/htmlcov"
        )
        
        for pattern in "${patterns[@]}"; do
            if [[ -e "$pattern" ]]; then
                local size
                size=$(du -sb "$pattern" 2>/dev/null | cut -f1 || echo "0")
                total_size=$((total_size + size))
            fi
        done
        
        if [[ $total_size -gt 0 ]]; then
            local human_size
            human_size=$(numfmt --to=iec --suffix=B "$total_size")
            log "Estimated space that would be freed: $human_size"
        fi
    fi
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
    -h, --help              Show this help message
    -a, --all               Clean everything (build, cache, data, logs)
    -b, --build             Clean build artifacts
    -c, --cache             Clean cache files
    -d, --data              Clean temporary data
    -l, --logs              Clean log files
    --dry-run               Show what would be cleaned without actually removing
    -v, --verbose           Enable verbose output

Examples:
    $0 --cache              # Clean only cache files
    $0 --build --cache      # Clean build artifacts and cache
    $0 --all                # Clean everything
    $0 --all --dry-run      # See what would be cleaned
EOF
}

# Main cleanup function
main() {
    log "Starting cleanup process..."
    log "Project root: $PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        warning "DRY RUN MODE - No files will actually be removed"
    fi
    
    # Check if any cleanup option is selected
    if [[ "$CLEAN_ALL" != true && "$CLEAN_BUILD" != true && "$CLEAN_CACHE" != true && "$CLEAN_DATA" != true && "$CLEAN_LOGS" != true ]]; then
        warning "No cleanup options specified. Use --help for available options."
        warning "Using --cache as default..."
        CLEAN_CACHE=true
    fi
    
    clean_build_artifacts
    clean_cache_files
    clean_temp_data
    clean_log_files
    
    show_dry_run_summary
    
    if [[ "$DRY_RUN" != true ]]; then
        success "Cleanup process completed successfully!"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -a|--all)
            CLEAN_ALL=true
            shift
            ;;
        -b|--build)
            CLEAN_BUILD=true
            shift
            ;;
        -c|--cache)
            CLEAN_CACHE=true
            shift
            ;;
        -d|--data)
            CLEAN_DATA=true
            shift
            ;;
        -l|--logs)
            CLEAN_LOGS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
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
