"""S3 storage implementation using pandas and s3fs."""

import boto3
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import s3fs
from .base import FileReader, VectorStorage


class S3FileReader(FileReader):
    """S3 file reader implementation using pandas and s3fs."""
    
    def __init__(self, **kwargs):
        """Initialize S3 file reader.
        
        Args:
            **kwargs: Additional arguments to pass to s3fs.S3FileSystem
        """
        self.fs = s3fs.S3FileSystem(**kwargs)
    
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read a CSV file from S3.
        
        Args:
            file_path: S3 path to the CSV file (e.g., 's3://bucket/file.csv')
            **kwargs: Additional arguments to pass to pandas.read_csv
            
        Returns:
            DataFrame containing the CSV data
        """
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"CSV file not found in S3: {file_path}")
        
        # Use s3fs directly with pandas
        with self.fs.open(str(file_path), 'r') as f:
            return pd.read_csv(f, **kwargs)
    
    def read_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read an Excel file from S3.
        
        Args:
            file_path: S3 path to the Excel file (e.g., 's3://bucket/file.xlsx')
            **kwargs: Additional arguments to pass to pandas.read_excel
            
        Returns:
            DataFrame containing the Excel data
        """
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"Excel file not found in S3: {file_path}")
        
        # Use s3fs directly with pandas  
        with self.fs.open(str(file_path), 'rb') as f:
            return pd.read_excel(f, **kwargs)
    
    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists in S3.
        
        Args:
            file_path: S3 path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            return self.fs.exists(str(file_path))
        except Exception:
            return False
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get the size of a file in bytes from S3.
        
        Args:
            file_path: S3 path to the file
            
        Returns:
            File size in bytes
        """
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"File not found in S3: {file_path}")
        
        try:
            info = self.fs.info(str(file_path))
            return info.get('size', 0)
        except Exception as e:
            raise RuntimeError(f"Failed to get file size for {file_path}: {e}")


class S3VectorStorage(VectorStorage):
    """S3 vector storage implementation using pandas and s3fs."""
    
    def __init__(self, **kwargs):
        """Initialize S3 vector storage.
        
        Args:
            **kwargs: Additional arguments to pass to s3fs.S3FileSystem
        """
        self.fs = s3fs.S3FileSystem(**kwargs)
    
    def save_parquet(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a parquet file in S3.
        
        Args:
            data: DataFrame to save
            file_path: S3 path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_parquet
        """
        file_path = str(file_path)
        self._ensure_bucket_exists(file_path)
        
        # Set default compression if not specified
        if 'compression' not in kwargs:
            kwargs['compression'] = 'snappy'
        
        # Use s3fs for S3 storage
        with self.fs.open(file_path, 'wb') as f:
            data.to_parquet(f, **kwargs)
    
    def save_csv(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a CSV file in S3.
        
        Args:
            data: DataFrame to save
            file_path: S3 path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_csv
        """
        file_path = str(file_path)
        self._ensure_bucket_exists(file_path)
        
        # Set default encoding if not specified
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        
        # Use s3fs for S3 storage
        with self.fs.open(file_path, 'w') as f:
            data.to_csv(f, index=False, **kwargs)
    
    def save_jsonl(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
        """Save a DataFrame to a JSONL file in S3.
        
        Args:
            data: DataFrame to save
            file_path: S3 path where to save the file
            **kwargs: Additional arguments to pass to pandas.to_json
        """
        file_path = str(file_path)
        self._ensure_bucket_exists(file_path)
        
        # Set default parameters for JSONL format
        if 'orient' not in kwargs:
            kwargs['orient'] = 'records'
        if 'lines' not in kwargs:
            kwargs['lines'] = True
        
        # Use s3fs for S3 storage
        with self.fs.open(file_path, 'w') as f:
            data.to_json(f, **kwargs)
    
    def directory_exists(self, directory_path: Union[str, Path]) -> bool:
        """Check if a directory exists in S3.
        
        Args:
            directory_path: S3 path to the directory
            
        Returns:
            True if directory exists, False otherwise
        """
        try:
            return self.fs.exists(str(directory_path))
        except Exception:
            return False
    
    def create_directory(self, directory_path: Union[str, Path]) -> None:
        """Create a directory in S3 if it doesn't exist.
        
        Args:
            directory_path: S3 path to the directory to create
        """
        # S3 doesn't have real directories, but we can create an empty object
        # to represent the directory structure
        try:
            dir_path = str(directory_path).rstrip('/') + '/'
            if not self.fs.exists(dir_path):
                self.fs.touch(dir_path)
        except Exception as e:
            # Directory creation is not critical for S3, so we just log the warning
            print(f"Warning: Could not create S3 directory {directory_path}: {e}")
    
    def _ensure_bucket_exists(self, file_path: str) -> None:
        """Ensure the S3 bucket exists.
        
        Args:
            file_path: S3 file path
        """
        # Extract bucket name from S3 path
        if file_path.startswith('s3://'):
            bucket_name = file_path.split('/')[2]
            try:
                # Check if bucket exists
                s3_client = boto3.client('s3')
                s3_client.head_bucket(Bucket=bucket_name)
            except Exception as e:
                raise RuntimeError(f"S3 bucket {bucket_name} does not exist or is not accessible: {e}")
