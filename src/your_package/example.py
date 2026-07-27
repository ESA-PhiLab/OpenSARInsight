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
Example module demonstrating coding standards and Apache 2.0 license header.

This module provides example functions following the specified coding guidelines.
"""

from typing import List, Optional


def process_data(input_data: List[str], filter_empty: bool = True) -> List[str]:
    """
    Process a list of strings with optional filtering.

    Args:
        input_data (List[str]): List of strings to process.
        filter_empty (bool): Whether to filter out empty strings. Defaults to True.

    Returns:
        List[str]: Processed list of strings.

    Raises:
        TypeError: If input_data is not a list.
        ValueError: If input_data is empty.
    """
    assert isinstance(input_data, list), 'input_data must be a list'
    assert len(input_data) > 0, 'input_data cannot be empty'
    assert isinstance(filter_empty, bool), 'filter_empty must be a boolean'
    
    processed_data = []
    
    for item in input_data:
        if isinstance(item, str):
            # Apply processing logic here
            processed_item = item.strip().lower()
            
            if filter_empty and not processed_item:
                continue
                
            processed_data.append(processed_item)
    
    return processed_data


class DataProcessor:
    """
    A class for processing data with various operations.
    
    Attributes:
        name (str): Name of the processor instance.
        config (Optional[dict]): Configuration dictionary.
    """
    
    def __init__(self, name: str, config: Optional[dict] = None) -> None:
        """
        Initialize the DataProcessor.

        Args:
            name (str): Name for this processor instance.
            config (Optional[dict]): Configuration dictionary. Defaults to None.
        """
        assert isinstance(name, str) and name.strip(), 'name must be a non-empty string'
        assert config is None or isinstance(config, dict), 'config must be a dictionary or None'
        
        self.name = name
        self.config = config or {}
    
    def transform_text(self, text: str) -> str:
        """
        Transform text according to processor settings.

        Args:
            text (str): Input text to transform.

        Returns:
            str: Transformed text.
        """
        assert isinstance(text, str), 'text must be a string'
        
        # Example transformation
        result = text.strip()
        
        if self.config.get('uppercase', False):
            result = result.upper()
        else:
            result = result.lower()
            
        return result
    
    def __str__(self) -> str:
        """Return string representation of the processor."""
        return f'DataProcessor(name={self.name!r})'
