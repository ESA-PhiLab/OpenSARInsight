#!/usr/bin/env python3
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

"""
Example utility script.

This script demonstrates the project's coding standards and Apache 2.0 license header.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def main() -> int:
    """
    Main entry point for the script.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description='Example utility script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        process_file(args.input, args.output, args.verbose)
        return 0
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1


def process_file(input_path: str, output_path: Optional[str] = None, verbose: bool = False) -> None:
    """
    Process a file according to the specified parameters.
    
    Args:
        input_path (str): Path to the input file.
        output_path (Optional[str]): Path to the output file. If None, prints to stdout.
        verbose (bool): Whether to enable verbose output.
        
    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the input file is empty.
    """
    assert isinstance(input_path, str) and input_path.strip(), 'input_path must be a non-empty string'
    assert output_path is None or isinstance(output_path, str), 'output_path must be a string or None'
    assert isinstance(verbose, bool), 'verbose must be a boolean'
    
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')
    
    if verbose:
        print(f'Processing file: {input_path}')
    
    # Example processing logic
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.strip():
        raise ValueError('Input file is empty')
    
    # Apply some transformation
    processed_content = content.upper()
    
    if output_path:
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        if verbose:
            print(f'Output written to: {output_path}')
    else:
        print(processed_content)


if __name__ == '__main__':
    sys.exit(main())
