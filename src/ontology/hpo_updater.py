import os
import re
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

class HPOUpdater:
    """
    Updates the local HPO ontology files by checking for newer versions.
    
    This class handles downloading the latest version of the HPO ontology
    from the official sources and storing it locally for use by the
    HPOGraphProvider.
    
    Attributes:
        data_dir: Directory where HPO data files are stored
        github_api_url: URL for the GitHub API to check releases
        jax_download_url: URL for direct download from JAX
        check_interval_days: How often to check for updates (in days)
        last_check: Timestamp of the last update check
    """
    
    def __init__(
        self,
        data_dir: str = "data/ontology",
        check_interval_days: int = 7,
        use_github: bool = True
    ):
        """
        Initialize the HPO updater.
        
        Args:
            data_dir: Directory where HPO data files are stored
            check_interval_days: How often to check for updates (in days)
            use_github: Whether to use GitHub as the primary source
        """
        self.data_dir = data_dir
        self.check_interval_days = check_interval_days
        self.use_github = use_github
        self.logger = logging.getLogger(__name__)
        
        # URLs for downloads
        self.github_api_url = "https://api.github.com/repos/obophenotype/human-phenotype-ontology/releases/latest"
        self.jax_base_url = "https://hpo.jax.org/app/data/ontology"
        
        # File paths
        self.data_dir_path = Path(data_dir)
        self.json_file = self.data_dir_path / "hp.json"
        self.obo_file = self.data_dir_path / "hp.obo"
        self.owl_file = self.data_dir_path / "hp.owl"
        self.version_file = self.data_dir_path / "version.json"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load version info
        self.version_info = self._load_version_info()
        self.last_check = self.version_info.get('last_check')
        
    def _load_version_info(self) -> Dict[str, Any]:
        """
        Load version information from the version file.
        
        Returns:
            Dictionary containing version information
        """
        if not self.version_file.exists():
            return {
                'version': None,
                'last_update': None,
                'last_check': None,
                'source': None
            }
            
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading version info: {str(e)}")
            return {
                'version': None,
                'last_update': None,
                'last_check': None,
                'source': None
            }
            
    def _save_version_info(self, version_info: Dict[str, Any]) -> None:
        """
        Save version information to the version file.
        
        Args:
            version_info: Dictionary containing version information
        """
        try:
            with open(self.version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving version info: {str(e)}")
    
    def should_check_update(self) -> bool:
        """
        Determine if it's time to check for updates.
        
        Returns:
            True if an update check should be performed, False otherwise
        """
        if self.last_check is None:
            return True
            
        try:
            last_check_date = datetime.fromisoformat(self.last_check)
            days_since_check = (datetime.now() - last_check_date).days
            return days_since_check >= self.check_interval_days
        except (ValueError, TypeError):
            return True
    
    def check_for_updates(self, force: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Check if a newer version of the HPO ontology is available.
        
        Args:
            force: Whether to force a check even if the interval hasn't elapsed
            
        Returns:
            Tuple of (update_available, latest_version)
        """
        if not force and not self.should_check_update():
            self.logger.info("Skipping update check (last check was recent)")
            return False, None
            
        # Update last check time
        self.version_info['last_check'] = datetime.now().isoformat()
        self._save_version_info(self.version_info)
        
        try:
            # Try GitHub first if configured
            if self.use_github:
                latest_version = self._check_github_version()
                if latest_version:
                    current_version = self.version_info.get('version')
                    if current_version != latest_version:
                        self.logger.info(f"New HPO version available: {latest_version}")
                        return True, latest_version
                    else:
                        self.logger.info(f"HPO is up to date (version {current_version})")
                        return False, current_version
            
            # Try JAX as fallback (or primary if GitHub not configured)
            latest_version = self._check_jax_version()
            if latest_version:
                current_version = self.version_info.get('version')
                if current_version != latest_version:
                    self.logger.info(f"New HPO version available: {latest_version}")
                    return True, latest_version
                else:
                    self.logger.info(f"HPO is up to date (version {current_version})")
                    return False, current_version
                    
            # If we got here, we couldn't determine the latest version
            self.logger.warning("Could not determine latest HPO version")
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking for updates: {str(e)}")
            return False, None
    
    def _check_github_version(self) -> Optional[str]:
        """
        Check the latest version available on GitHub.
        
        Returns:
            Latest version string or None if not found
        """
        try:
            response = requests.get(self.github_api_url)
            response.raise_for_status()
            
            release_data = response.json()
            tag_name = release_data.get('tag_name', '')
            
            # Extract version from tag (typically v1.2.3 format)
            match = re.search(r'v?(\d+\.\d+\.\d+)', tag_name)
            if match:
                return match.group(1)
                
            return tag_name
            
        except Exception as e:
            self.logger.warning(f"Could not check GitHub version: {str(e)}")
            return None
    
    def _check_jax_version(self) -> Optional[str]:
        """
        Check the latest version available on JAX.
        
        Returns:
            Latest version string or None if not found
        """
        try:
            # Try to determine version from OBO file header
            response = requests.get(f"{self.jax_base_url}/hp.obo", stream=True)
            response.raise_for_status()
            
            # Read the first few lines to extract version
            for i, line in enumerate(response.iter_lines(decode_unicode=True)):
                if i > 100:  # Stop after checking the first 100 lines
                    break
                    
                if line.startswith('data-version:'):
                    version = line.split('data-version:')[1].strip()
                    return version
                    
            return datetime.now().strftime("%Y-%m-%d")  # Use current date as fallback
            
        except Exception as e:
            self.logger.warning(f"Could not check JAX version: {str(e)}")
            return None
    
    def update(self, format_type: str = "json") -> bool:
        """
        Download and update the local HPO files.
        
        Args:
            format_type: Format to download ('json', 'obo', 'owl', or 'all')
            
        Returns:
            True if update was successful, False otherwise
        """
        update_available, latest_version = self.check_for_updates()
        
        if not update_available and self.version_info.get('version') is not None:
            self.logger.info("No update needed, HPO is already up to date")
            return True
            
        if latest_version is None:
            latest_version = datetime.now().strftime("%Y-%m-%d")
            
        # Determine which formats to download
        formats_to_download = []
        if format_type == "all":
            formats_to_download = ["json", "obo", "owl"]
        else:
            formats_to_download = [format_type]
            
        # Download each requested format
        success = True
        for fmt in formats_to_download:
            if self.use_github:
                fmt_success = self._download_from_github(fmt, latest_version)
                if not fmt_success:
                    # Fall back to JAX if GitHub fails
                    fmt_success = self._download_from_jax(fmt)
            else:
                fmt_success = self._download_from_jax(fmt)
                
            success = success and fmt_success
            
        if success:
            # Update version info
            self.version_info['version'] = latest_version
            self.version_info['last_update'] = datetime.now().isoformat()
            self.version_info['source'] = "github" if self.use_github else "jax"
            self._save_version_info(self.version_info)
            
            self.logger.info(f"Successfully updated HPO to version {latest_version}")
            return True
        else:
            self.logger.error("Failed to update HPO")
            return False
    
    def _download_from_github(self, format_type: str, version: str) -> bool:
        """
        Download HPO files from GitHub.
        
        Args:
            format_type: Format to download ('json', 'obo', or 'owl')
            version: Version to download
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            # Get the latest release details
            response = requests.get(self.github_api_url)
            response.raise_for_status()
            
            release_data = response.json()
            assets = release_data.get('assets', [])
            
            # Find the correct asset
            target_filename = f"hp.{format_type}"
            for asset in assets:
                asset_name = asset.get('name', '')
                if asset_name.lower() == target_filename.lower():
                    download_url = asset.get('browser_download_url')
                    
                    # Download the file
                    self.logger.info(f"Downloading {target_filename} from GitHub...")
                    return self._download_file(download_url, getattr(self, f"{format_type}_file"))
            
            self.logger.warning(f"Could not find {target_filename} in GitHub release")
            return False
            
        except Exception as e:
            self.logger.error(f"Error downloading from GitHub: {str(e)}")
            return False
    
    def _download_from_jax(self, format_type: str) -> bool:
        """
        Download HPO files from JAX.
        
        Args:
            format_type: Format to download ('json', 'obo', or 'owl')
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            # Construct download URL
            download_url = f"{self.jax_base_url}/hp.{format_type}"
            target_file = getattr(self, f"{format_type}_file")
            
            # Download the file
            self.logger.info(f"Downloading {format_type} file from JAX...")
            return self._download_file(download_url, target_file)
            
        except Exception as e:
            self.logger.error(f"Error downloading from JAX: {str(e)}")
            return False
    
    def _download_file(self, url: str, target_path: Path) -> bool:
        """
        Download a file from a URL to a target path.
        
        Args:
            url: URL to download from
            target_path: Path to save the file to
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save the file
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            self.logger.info(f"Successfully downloaded to {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            return False 