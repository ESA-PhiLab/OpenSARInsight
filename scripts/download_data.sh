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

# Data download and preprocessing script
# This script handles downloading datasets and preparing them for analysis

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
DATA_DIR="$PROJECT_ROOT/data"
DOWNLOAD_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"

# Default options
FORCE_DOWNLOAD=false
SKIP_PROCESSING=false
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

# Check if required tools are available
check_dependencies() {
    log "Checking dependencies..."
    
    local required_tools=("curl" "python3")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        error "Please install them before running this script."
        exit 1
    fi
    
    success "All dependencies are available"
}

# Create necessary directories
create_directories() {
    log "Creating data directories..."
    
    local dirs=(
        "$DOWNLOAD_DIR"
        "$PROCESSED_DIR"
        "$DATA_DIR/external"
        "$DATA_DIR/interim"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            verbose "Created directory: $dir"
        fi
    done
    
    success "Data directories ready"
}

# Download a file with curl
download_file() {
    local url="$1"
    local output_path="$2"
    local description="${3:-file}"
    
    log "Downloading $description..."
    verbose "URL: $url"
    verbose "Output: $output_path"
    
    # Check if file already exists
    if [[ -f "$output_path" && "$FORCE_DOWNLOAD" != true ]]; then
        warning "File already exists: $output_path (use --force to re-download)"
        return 0
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"
    
    # Download with curl
    if curl -L --fail --progress-bar -o "$output_path" "$url"; then
        success "Downloaded $description successfully"
    else
        error "Failed to download $description from $url"
        return 1
    fi
}

# Download sample datasets
download_datasets() {
    log "Downloading datasets..."
    
    # Example: Download a sample CSV file
    # Replace these with actual dataset URLs
    local datasets=(
        "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv|iris.csv|Iris dataset"
        "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv|tips.csv|Tips dataset"
    )
    
    for dataset_info in "${datasets[@]}"; do
        IFS='|' read -r url filename description <<< "$dataset_info"
        download_file "$url" "$DOWNLOAD_DIR/$filename" "$description"
    done
    
    # Run Python script to download from Hugging Face if available
    if [[ -f "$PROJECT_ROOT/pyscripts/download_hf_datasets.py" ]]; then
        log "Running Hugging Face dataset download script..."
        cd "$PROJECT_ROOT"
        python pyscripts/download_hf_datasets.py || warning "HF download script failed"
    fi
}

# Process downloaded data
process_data() {
    if [[ "$SKIP_PROCESSING" == true ]]; then
        log "Skipping data processing"
        return 0
    fi
    
    log "Processing downloaded data..."
    
    # Check if there are any files to process
    if [[ ! "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ]]; then
        warning "No files found in download directory: $DOWNLOAD_DIR"
        return 0
    fi
    
    # Basic file validation and organization
    for file in "$DOWNLOAD_DIR"/*; do
        if [[ -f "$file" ]]; then
            local filename
            filename=$(basename "$file")
            local extension="${filename##*.}"
            
            verbose "Processing file: $filename"
            
            case "$extension" in
                csv)
                    # Validate CSV files
                    if head -1 "$file" | grep -q ","; then
                        verbose "Valid CSV file: $filename"
                        # Copy to processed directory with timestamp
                        cp "$file" "$PROCESSED_DIR/${filename%.*}_$(date +%Y%m%d).csv"
                    else
                        warning "Invalid CSV format: $filename"
                    fi
                    ;;
                json)
                    # Validate JSON files
                    if python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
                        verbose "Valid JSON file: $filename"
                        cp "$file" "$PROCESSED_DIR/${filename%.*}_$(date +%Y%m%d).json"
                    else
                        warning "Invalid JSON format: $filename"
                    fi
                    ;;
                *)
                    verbose "Unknown file type: $filename"
                    ;;
            esac
        fi
    done
    
    success "Data processing completed"
}

# Generate data summary
generate_summary() {
    log "Generating data summary..."
    
    local summary_file="$DATA_DIR/data_summary.txt"
    
    cat > "$summary_file" << EOF
Data Summary Report
Generated on: $(date)

Raw Data Directory: $DOWNLOAD_DIR
Processed Data Directory: $PROCESSED_DIR

Raw Files:
EOF
    
    if [[ -d "$DOWNLOAD_DIR" && "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ]]; then
        find "$DOWNLOAD_DIR" -type f -exec ls -lh {} \; >> "$summary_file"
    else
        echo "No raw files found" >> "$summary_file"
    fi
    
    echo "" >> "$summary_file"
    echo "Processed Files:" >> "$summary_file"
    
    if [[ -d "$PROCESSED_DIR" && "$(ls -A "$PROCESSED_DIR" 2>/dev/null)" ]]; then
        find "$PROCESSED_DIR" -type f -exec ls -lh {} \; >> "$summary_file"
    else
        echo "No processed files found" >> "$summary_file"
    fi
    
    success "Data summary generated: $summary_file"
    
    if [[ "$VERBOSE" == true ]]; then
        log "Summary contents:"
        cat "$summary_file"
    fi
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
    -h, --help              Show this help message
    -f, --force             Force re-download of existing files
    --skip-processing       Skip data processing step
    -v, --verbose           Enable verbose output

This script downloads datasets and prepares them for analysis.

Examples:
    $0                      # Download and process data
    $0 --force              # Force re-download all files
    $0 --skip-processing    # Download only, skip processing
EOF
}

# Main function
main() {
    log "Starting data download and processing..."
    log "Data directory: $DATA_DIR"
    
    check_dependencies
    create_directories
    download_datasets
    process_data
    generate_summary
    
    success "Data download and processing completed successfully!"
    log ""
    log "Data locations:"
    log "  Raw data: $DOWNLOAD_DIR"
    log "  Processed data: $PROCESSED_DIR"
    log "  Summary: $DATA_DIR/data_summary.txt"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -f|--force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --skip-processing)
            SKIP_PROCESSING=true
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
