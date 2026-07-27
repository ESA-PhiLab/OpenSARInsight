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
Dataset download utility for Hugging Face Hub.

This script downloads datasets or models from the Hugging Face Hub using the 
`huggingface_hub` library with robust error handling and retry mechanisms.
"""

# pip install huggingface_hub
import os
import time
from huggingface_hub import snapshot_download

# Set the environment variable
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def download_dataset(repo_id: str, repo_type: str = 'dataset', local_dir: str = './', 
                    max_retries: int = 5, retry_delay: int = 5) -> str:
    """
    Downloads a dataset or model from the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        repo_type (str): The type of repository. Defaults to 'dataset'.
        local_dir (str): Optional local directory to store the files. If None, uses the default cache directory.
        max_retries (int): Maximum number of retries in case of failure. Defaults to 5.
        retry_delay (int): Delay (in seconds) between retries. Defaults to 5 seconds.

    Returns:
        str: Path to the downloaded repository.
    
    Raises:
        Exception: If download fails after maximum retries.
    """
    assert isinstance(repo_id, str) and repo_id.strip(), 'repo_id must be a non-empty string'
    assert isinstance(repo_type, str) and repo_type.strip(), 'repo_type must be a non-empty string'
    assert isinstance(local_dir, str), 'local_dir must be a string'
    assert isinstance(max_retries, int) and max_retries > 0, 'max_retries must be a positive integer'
    assert isinstance(retry_delay, int) and retry_delay >= 0, 'retry_delay must be a non-negative integer'
    retries = 0
    while retries < max_retries:
        try:
            print(f'Starting download from repo: {repo_id} (type: {repo_type})')
            download_path = snapshot_download(
                repo_id=repo_id, 
                repo_type=repo_type, 
                force_download=True,
                max_workers=1,
                local_dir=local_dir
            )
            print(f'Download complete. Files are located at: {download_path}')
            return download_path
        except Exception as e:
            retries += 1
            print(f'Error occurred while downloading {repo_id}: {e}')
            if retries < max_retries:
                print(f'Retrying download ({retries}/{max_retries}) after {retry_delay} seconds...')
                time.sleep(retry_delay)
            else:
                print(f'Exceeded maximum retries for {repo_id}. Skipping...')
                raise

if __name__ == '__main__':
    print('Starting download script...')
    
    # Define repository details
    # List of datasets to download
    repo_ids = [
        'distilbert-base-uncased',  # Example pre-trained model
        'huggingface/datasets-examples',  # Example dataset repository
        # Add more repositories as needed
    ]
    
    # Create a directory for downloaded files
    local_directory = '.'
    
    # Download repositories one at a time
    for repo_id in repo_ids:
        try:
            download_dataset(repo_id=repo_id, repo_type='dataset', local_dir=local_directory)
        except Exception as e:
            print(f'Skipping {repo_id} due to an error: {e}')