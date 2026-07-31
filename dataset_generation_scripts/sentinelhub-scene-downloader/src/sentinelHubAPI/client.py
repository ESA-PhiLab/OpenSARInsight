"""
Main client module for Copernicus Data Space Ecosystem.
Provides high-level interface combining query, download, and quicklook functionality.
"""
from .auth import CDSEAuth
from .query import CDSEQuery
from .downloader import CDSEDownloader
from .quicklook import CDSEQuicklook


class CopernicusDataSpace:
    """Client for Copernicus Data Space Ecosystem operations."""
    
    def __init__(self, base_url: str, username: str, password: str):
        """Initialize client with configuration and credentials."""
        self.base_url = base_url
        
        # Initialize components
        self.auth = CDSEAuth(username, password)
        self.query = CDSEQuery(base_url, self.auth)
        self.downloader = CDSEDownloader(self.auth)
        self.quicklook = CDSEQuicklook(self.auth)
    
    def search_product(self, product_name: str): # type: ignore
        """Search for a product by name."""
        return self.query.query_by_name(product_name) # type: ignore
    
    def download_product(self, product_id: str, download_dir: str = "./downloads"):
        """Download a product by ID."""
        return self.downloader.download_product(product_id, download_dir)
    
    def search_and_download(self, product_name: str, download_dir: str = "./downloads"):
        """Search for a product and download it if found."""
        # Search for the product
        result = self.search_product(product_name) # type: ignore
        
        if result and result.get('value'): # type: ignore
            products = result['value'] # type: ignore
            if products:
                product = products[0]  # Get first match # type: ignore
                product_id = product['Id'] # type: ignore
                print(f"\nFound product: {product['Name']}")
                print(f"Product ID: {product_id}")
                
                # Download the product
                return self.download_product(product_id, download_dir) # type: ignore
            else:
                print("No products found with that name")
                return False
        else:
            print("Search failed or returned no results")
            return False
    
    def get_product_info(self, product_id: str): # type: ignore
        """Get detailed information about a product."""
        return self.query.get_product_info(product_id) # type: ignore
    
    def get_quicklook_info(self, product_id: str):  # type: ignore
        """Get quicklook information for a product."""
        return self.quicklook.get_quicklook_info(product_id)  # type: ignore
    
    def download_quicklook(self, quicklook_id: str, download_dir: str = "./quicklooks", filename: str = None):  # type: ignore
        """Download a quicklook image."""
        return self.quicklook.download_quicklook(quicklook_id, download_dir, filename)
    
    def view_and_download_quicklook(self, product_name: str, download_dir: str = "./quicklooks"):
        """Search for a product and download its quicklook if available."""
        # Search for the product
        result = self.search_product(product_name)  # type: ignore
        
        if result and result.get('value'):  # type: ignore
            products = result['value']  # type: ignore
            if products:
                product = products[0]  # Get first match  # type: ignore
                product_id = product['Id']  # type: ignore
                print(f"\nFound product: {product['Name']}")
                
                # Get quicklook info
                quicklooks = self.get_quicklook_info(product_id)  # type: ignore
                
                if quicklooks:
                    print(f"Found {len(quicklooks)} quicklook(s)")  # type: ignore
                    for i, ql in enumerate(quicklooks):  # type: ignore
                        print(f"  {i+1}. {ql['name']} (ID: {ql['id']})")
                        
                        # Download the first quicklook
                        if i == 0:
                            filename = f"{product['Name']}_quicklook.jpg"
                            return self.download_quicklook(ql['id'], download_dir, filename)  # type: ignore
                else:
                    print("No quicklooks available for this product")
                    return False
            else:
                print("No products found with that name")
                return False
        else:
            print("Search failed or returned no results")
            return False
