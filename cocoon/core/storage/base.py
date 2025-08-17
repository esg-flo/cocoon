"""Abstract storage interfaces for the Database Vectorizer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd


class FileReader(ABC):
    """Abstract base class for file reading operations."""
    
    @abstractmethod
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read a CSV file and return a pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv
            
        Returns:
            DataFrame containing the CSV data
        """
        pass
    
    @abstractmethod
    def read_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read an Excel file and return a pandas DataFrame.
        
        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments to pass to pandas.read_excel
            
        Returns:
            DataFrame containing the Excel data
        """
        pass
    
    @abstractmethod
    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        pass


class VectorStorage(ABC):
    """Abstract base class for vector storage operations."""
    
    @abstractmethod
    def save_parquet(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a parquet file.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_parquet
        """
        pass
    
    @abstractmethod
    def save_csv(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_csv
        """
        pass
    
    @abstractmethod
    def save_jsonl(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a JSONL file.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_json
        """
        pass
    
    @abstractmethod
    def directory_exists(self, directory_path: Union[str, Path]) -> bool:
        """Check if a directory exists.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory exists, False otherwise
        """
        pass
    
    @abstractmethod
    def create_directory(self, directory_path: Union[str, Path]) -> None:
        """Create a directory if it doesn't exist.
        
        Args:
            directory_path: Path to the directory to create
        """
        pass
