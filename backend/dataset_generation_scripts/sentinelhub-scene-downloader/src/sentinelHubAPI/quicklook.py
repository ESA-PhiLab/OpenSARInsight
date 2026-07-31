"""
Quicklook module for Copernicus Data Space Ecosystem.
Handles quicklook image retrieval and viewing.
"""
import os
import requests
from .auth import CDSEAuth

class CDSEQuicklook:
    """Handles quicklook operations for Copernicus Data Space Ecosystem."""
    
    def __init__(self, auth: CDSEAuth):
        """Initialize quicklook handler with authentication."""
        self.auth = auth
    
    def get_quicklook_info(self, product_id: str):  # type: ignore
        """Get quicklook information for a product."""
        headers = self.auth.get_auth_headers()  # type: ignore
        if not headers.get("Authorization"):  # type: ignore
            print("Failed to get authentication token")
            return None
        
        # Query for assets related to the product
        assets_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/Assets"
        
        try:
            response = requests.get(assets_url, headers=headers)  # type: ignore
            response.raise_for_status()
            assets_data = response.json()
            
            # Find quicklook assets
            quicklooks = []
            if 'value' in assets_data:
                for asset in assets_data['value']:
                    if asset.get('Category') == 'QUICKLOOK':
                        quicklooks.append({  # type: ignore
                            'id': asset.get('Id'),
                            'name': asset.get('Name'),
                            'download_url': asset.get('DownloadLink')
                        })
            
            return quicklooks  # type: ignore
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting quicklook info: {e}")
            return None
    
    def download_quicklook(self, quicklook_id: str, download_dir: str = "./quicklooks", filename: str = None):  # type: ignore
        """Download a quicklook by ID."""
        headers = self.auth.get_auth_headers()  # type: ignore
        if not headers.get("Authorization"):  # type: ignore
            print("Failed to get authentication token")
            return False
        
        # Construct quicklook download URL as per documentation
        quicklook_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Assets({quicklook_id})/$value"
        
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        # Set filename
        if not filename:
            filename = f"quicklook_{quicklook_id}.jpg"
        
        filepath = os.path.join(download_dir, filename)
        
        try:
            print(f"Downloading quicklook from: {quicklook_url}")
            
            # Create session and update headers
            session = requests.Session()
            session.headers.update(headers)  # type: ignore
            
            # Download quicklook
            response = session.get(quicklook_url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Quicklook downloaded: {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading quicklook: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
