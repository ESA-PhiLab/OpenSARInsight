"""
Query module for Copernicus Data Space Ecosystem.
Handles product search and metadata retrieval.
"""
import json
import requests
from .auth import CDSEAuth


class CDSEQuery:
    """Handles product queries for Copernicus Data Space Ecosystem."""
    
    def __init__(self, base_url: str, auth: CDSEAuth):
        """Initialize query handler with base URL and authentication."""
        self.base_url = base_url
        self.auth = auth
    
    def query_by_name(self, product_name: str) -> dict: # type: ignore
        """Query product by name using OData API."""
        # Get authentication headers
        headers = self.auth.get_auth_headers() # type: ignore
        if not headers.get("Authorization"): # type: ignore
            print("Failed to get authentication token")
            return None # type: ignore
        
        # Construct OData query with properly quoted product name
        quoted_product_name = f"'{product_name}'"
        full_url = self.base_url + quoted_product_name
        
        # Ensure HTTPS
        if not full_url.startswith('https://'):
            full_url = full_url.replace('http://', 'https://')
        
        try:
            # Make authenticated request
            response = requests.get(full_url, headers=headers) # type: ignore
            response.raise_for_status()
            
            data = response.json()
            
            print(f"Request URL: {full_url}")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {full_url}: {e}")
            return None # type: ignore
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None # type: ignore
    
    def get_product_info(self, product_id: str) -> dict: # type: ignore
        """Get detailed product information by ID."""
        headers = self.auth.get_auth_headers() # type: ignore
        if not headers.get("Authorization"): # type: ignore
            print("Failed to get authentication token")
            return None # type: ignore
        
        info_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})"
        
        try:
            response = requests.get(info_url, headers=headers) # type: ignore
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting product info: {e}")
            return None # type: ignore
