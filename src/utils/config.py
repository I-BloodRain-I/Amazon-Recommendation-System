import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for the recommendation system."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'models.sbert_model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(__file__).parent.parent.parent / self.get('paths.data_dir')
    
    @property
    def database_dir(self) -> Path:
        """Get database directory path."""
        return Path(__file__).parent.parent.parent / self.get('paths.database_dir')
    
    @property
    def reviews_file(self) -> str:
        """Get reviews filename."""
        return self.get('paths.reviews_file')
    
    @property
    def index_name(self) -> str:
        """Get index base name."""
        return self.get('paths.index_name')
    
    @property
    def index_ext(self) -> str:
        """Get index file extension."""
        return self.get('paths.index_ext')
    
    @property
    def metadata_ext(self) -> str:
        """Get metadata file extension."""
        return self.get('paths.metadata_ext')


# Singleton instance
config = Config()
