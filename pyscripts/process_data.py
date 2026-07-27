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
Data processing utility.

This script provides utilities for processing and analyzing datasets
following the project's coding standards.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from various file formats.
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    """
    assert isinstance(file_path, str) and file_path.strip(), 'file_path must be a non-empty string'
    
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f'File not found: {file_path}')
    
    suffix = file_path_obj.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(file_path)
    elif suffix == '.json':
        return pd.read_json(file_path)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f'Unsupported file format: {suffix}')


def analyze_dataset(df: pd.DataFrame, output_format: str = 'dict') -> Union[Dict, str]:
    """
    Perform basic analysis on a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze.
        output_format (str): Output format ('dict' or 'json'). Defaults to 'dict'.
        
    Returns:
        Union[Dict, str]: Analysis results.
    """
    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
    assert output_format in ['dict', 'json'], 'output_format must be either "dict" or "json"'
    
    analysis = {
        'shape': {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1])
        },
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': {
            'total_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
        }
    }
    
    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        analysis['numerical_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        analysis['categorical_summary'] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10)
            analysis['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.to_dict()
            }
    
    return json.dumps(analysis, indent=2) if output_format == 'json' else analysis


def clean_dataset(df: pd.DataFrame, 
                  drop_duplicates: bool = True,
                  fill_numeric_na: str = 'mean',
                  fill_categorical_na: str = 'mode') -> pd.DataFrame:
    """
    Clean a dataset by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Dataset to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Defaults to True.
        fill_numeric_na (str): Strategy for filling numeric NAs ('mean', 'median', 'zero'). 
                               Defaults to 'mean'.
        fill_categorical_na (str): Strategy for filling categorical NAs ('mode', 'unknown'). 
                                   Defaults to 'mode'.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
    assert isinstance(drop_duplicates, bool), 'drop_duplicates must be a boolean'
    assert fill_numeric_na in ['mean', 'median', 'zero'], 'fill_numeric_na must be "mean", "median", or "zero"'
    assert fill_categorical_na in ['mode', 'unknown'], 'fill_categorical_na must be "mode" or "unknown"'
    
    df_cleaned = df.copy()
    
    # Drop duplicates
    if drop_duplicates:
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        dropped_rows = initial_rows - len(df_cleaned)
        if dropped_rows > 0:
            print(f'Dropped {dropped_rows} duplicate rows')
    
    # Handle numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            if fill_numeric_na == 'mean':
                fill_value = df_cleaned[col].mean()
            elif fill_numeric_na == 'median':
                fill_value = df_cleaned[col].median()
            else:  # zero
                fill_value = 0
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            print(f'Filled {col} NAs with {fill_numeric_na}: {fill_value}')
    
    # Handle categorical columns
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            if fill_categorical_na == 'mode':
                mode_value = df_cleaned[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
            else:  # unknown
                fill_value = 'unknown'
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            print(f'Filled {col} NAs with: {fill_value}')
    
    return df_cleaned


def save_dataset(df: pd.DataFrame, output_path: str, format_type: Optional[str] = None) -> None:
    """
    Save a dataset to file.
    
    Args:
        df (pd.DataFrame): Dataset to save.
        output_path (str): Output file path.
        format_type (Optional[str]): File format ('csv', 'json', 'parquet'). 
                                     If None, infers from extension.
    """
    assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
    assert isinstance(output_path, str) and output_path.strip(), 'output_path must be a non-empty string'
    
    output_path_obj = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format_type is None:
        format_type = output_path_obj.suffix.lower().lstrip('.')
    
    if format_type == 'csv':
        df.to_csv(output_path, index=False)
    elif format_type == 'json':
        df.to_json(output_path, orient='records', indent=2)
    elif format_type == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f'Unsupported output format: {format_type}')
    
    print(f'Dataset saved to: {output_path}')


def main() -> int:
    """
    Main entry point for the data processing script.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description='Data processing utility',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Input dataset file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional)'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Perform dataset analysis'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean the dataset'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'parquet'],
        help='Output format (if not specified, infers from extension)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        if args.verbose:
            print(f'Loading dataset from: {args.input_file}')
        
        df = load_dataset(args.input_file)
        
        if args.verbose:
            print(f'Loaded dataset with shape: {df.shape}')
        
        # Analyze dataset
        if args.analyze:
            print('Dataset Analysis:')
            analysis = analyze_dataset(df, output_format='json')
            print(analysis)
        
        # Clean dataset
        if args.clean:
            if args.verbose:
                print('Cleaning dataset...')
            
            df = clean_dataset(df)
            
            if args.verbose:
                print(f'Cleaned dataset shape: {df.shape}')
        
        # Save dataset
        if args.output:
            save_dataset(df, args.output, args.format)
        elif args.clean:
            # If cleaning but no output specified, save to processed version
            input_path = Path(args.input_file)
            output_path = input_path.parent / 'processed' / f'{input_path.stem}_cleaned{input_path.suffix}'
            save_dataset(df, str(output_path))
        
        return 0
        
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
