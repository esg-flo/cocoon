"""Local file system storage implementation."""

import os
from pathlib import Path
from typing import Union
import pandas as pd
from .base import FileReader, VectorStorage


class LocalFileReader(FileReader):
    """Local file system file reader implementation."""
    
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read a CSV file from local file system.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv
            
        Returns:
            DataFrame containing the CSV data
        """
        file_path = Path(file_path)
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        return pd.read_csv(file_path, **kwargs)
    
    def read_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read an Excel file from local file system.
        
        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments to pass to pandas.read_excel
            
        Returns:
            DataFrame containing the Excel data
        """
        file_path = Path(file_path)
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        return pd.read_excel(file_path, **kwargs)
    
    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists in local file system.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        return Path(file_path).is_file()
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get the size of a file in bytes from local file system.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        file_path = Path(file_path)
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return file_path.stat().st_size


class LocalVectorStorage(VectorStorage):
    """Local file system vector storage implementation."""
    
    def save_parquet(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a parquet file in local file system.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_parquet
        """
        file_path = Path(file_path)
        self._ensure_directory_exists(file_path.parent)
        
        # Set default compression if not specified
        if 'compression' not in kwargs:
            kwargs['compression'] = 'snappy'
        
        data.to_parquet(file_path, **kwargs)
    
    def save_csv(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a CSV file in local file system.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_csv
        """
        file_path = Path(file_path)
        self._ensure_directory_exists(file_path.parent)
        
        # Set default encoding if not specified
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        
        data.to_csv(file_path, index=False, **kwargs)
    
    def save_jsonl(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a JSONL file in local file system.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_json
        """
        file_path = Path(file_path)
        self._ensure_directory_exists(file_path.parent)
        
        # Set default parameters for JSONL format
        if 'orient' not in kwargs:
            kwargs['orient'] = 'records'
        if 'lines' not in kwargs:
            kwargs['lines'] = True
        
        data.to_json(file_path, **kwargs)
    
    def directory_exists(self, directory_path: Union[str, Path]) -> bool:
        """Check if a directory exists in local file system.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory exists, False otherwise
        """
        return Path(directory_path).is_dir()
    
    def create_directory(self, directory_path: Union[str, Path]) -> None:
        """Create a directory in local file system if it doesn't exist.
        
        Args:
            directory_path: Path to the directory to create
        """
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    
    def _ensure_directory_exists(self, directory_path: Path) -> None:
        """Ensure a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to the directory
        """
        if directory_path and not directory_path.exists():
            self.create_directory(directory_path)
