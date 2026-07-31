"""
Download module for Copernicus Data Space Ecosystem.
Handles product downloading with progress tracking.
"""
import os
import requests
from .auth import CDSEAuth
from .query import CDSEQuery


class CDSEDownloader:
    """Handles product downloads for Copernicus Data Space Ecosystem."""
    
    def __init__(self, auth: CDSEAuth):
        """Initialize downloader with authentication."""
        self.auth = auth
    
    def download_product(self, product_id: str, download_dir: str = "./downloads") -> bool:
        """Download a product by ID to specified directory."""
        # Get authentication headers
        headers = self.auth.get_auth_headers() # type: ignore
        if not headers.get("Authorization"): # type: ignore
            print("Failed to get authentication token")
            return False
        
        # Get product info first
        query = CDSEQuery("", self.auth)  # Empty base_url since we're using direct ID
        product_info = query.get_product_info(product_id) # type: ignore
        
        if not product_info:
            print("Failed to get product information")
            return False
        
        # Prepare download - use the correct download endpoint from documentation
        product_name = product_info.get('Name', f'product_{product_id}') # type: ignore
        # Format filename as ProductName.SAFE.zip
        if product_name.endswith('.SAFE'):  # type: ignore
            filename = f"{product_name}.zip"
        else:
            filename = f"{product_name}.SAFE.zip"
        
        # Use the correct download endpoint as per CDSE documentation
        download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        filepath = os.path.join(download_dir, filename) # type: ignore
        
        print(f"Starting download of {filename}...")
        print(f"File size: {product_info.get('ContentLength', 'Unknown')} bytes") # type: ignore
        
        try:
            # Get fresh authentication headers for download
            download_headers = self.auth.get_auth_headers()  # type: ignore
            if not download_headers.get("Authorization"):  # type: ignore
                print("Failed to get fresh authentication token")
                return False
            
            print(f"Download URL: {download_url}")
            print("Creating  session for download...")
            
            # Create session and update headers as per CDSE documentation
            session = requests.Session() # type: ignore
            session.headers.update(download_headers) # type: ignore
            
            # Download with progress tracking using session
            with session.get(download_url, stream=True) as response: # type: ignore
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress indicator
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", 
                                      end='', flush=True)
            
            print(f"\nDownload completed: {filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading product: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
