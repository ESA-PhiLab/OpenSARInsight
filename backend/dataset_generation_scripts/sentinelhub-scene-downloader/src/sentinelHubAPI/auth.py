"""
Authentication module for Copernicus Data Space Ecosystem.
Handles token generation and authentication operations.
"""
import requests


class CDSEAuth:
    """Handles authentication for Copernicus Data Space Ecosystem."""
    
    def __init__(self, username: str, password: str):
        """Initialize authentication with user credentials."""
        self.username = username
        self.password = password
        self._token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    def get_access_token(self) -> str:
        """Get access token using username/password authentication."""
        data = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "client_id": "cdse-public"
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        try:
            response = requests.post(self._token_url, data=data, headers=headers)
            response.raise_for_status()
            token_info = response.json()
            return token_info["access_token"]
        except requests.exceptions.RequestException as e:
            print(f"Error getting CDSE token: {e}")
            return None # type: ignore
    
    def get_auth_headers(self) -> dict:  # type: ignore
        """Get authentication headers with Bearer token."""
        token = self.get_access_token()
        if token:
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }  # type: ignore
        return {} # type: ignore
