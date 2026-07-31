"""
Copernicus Datfrom .client import CopernicusDataSpace
from .auth import CDSEAuth
from .query import CDSEQuery
from .downloader import CDSEDownloader
from .quicklook import CDSEQuicklook

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "CopernicusDataSpace",
    "CDSEAuth", 
    "CDSEQuery",
    "CDSEDownloader",
    "CDSEQuicklook"
]tem Python Client

A modular Python client for interacting with the Copernicus Data Space Ecosystem,
providing authentication, product search, and download capabilities.

Example usage:
    from sentinelHubAPI import CopernicusDataSpace
    
    client = CopernicusDataSpace(
        base_url="https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name%20eq%20",
        username="your_username",
        password="your_password"
    )
    
    # Search and download
    client.search_and_download("S1A_IW_SLC__1SDV_20200407T052956_20200407T053026_032017_03B2E2_18FA.SAFE")
"""

from .client import CopernicusDataSpace
from .auth import CDSEAuth
from .query import CDSEQuery
from .downloader import CDSEDownloader

__version__ = "1.0.0"
__author__ = "Abdulhameed Yunusa"

__all__ = [
    "CopernicusDataSpace",
    "CDSEAuth", 
    "CDSEQuery",
    "CDSEDownloader"
]
