"""Unit tests for S3 storage implementation."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from cocoon.core.storage.s3 import S3FileReader, S3VectorStorage


class TestS3FileReader:
    """Test S3FileReader class."""
    
    def test_init_with_kwargs(self):
        """Test S3FileReader initialization with kwargs."""
        with patch('cocoon.core.storage.s3.s3fs.S3FileSystem') as mock_s3fs:
            mock_fs = MagicMock()
            mock_s3fs.return_value = mock_fs
            
            reader = S3FileReader(profile_name='test-profile', region='us-east-1')
            
            mock_s3fs.assert_called_once_with(profile_name='test-profile', region='us-east-1')
            assert reader.fs == mock_fs
    
    def test_read_csv_success(self):
        """Test successful CSV file reading from S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        
        # Mock the file handle returned by fs.open()
        mock_file_handle = MagicMock()
        mock_fs.open.return_value.__enter__.return_value = mock_file_handle
        
        test_data = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with patch('pandas.read_csv', return_value=test_data) as mock_read_csv:
            reader = S3FileReader()
            reader.fs = mock_fs
            
            result = reader.read_csv('s3://bucket/file.csv')
            
            # Verify that fs.open was called correctly
            mock_fs.open.assert_called_once_with('s3://bucket/file.csv', 'r')
            
            # Verify that pandas.read_csv was called with the file handle
            mock_read_csv.assert_called_once_with(mock_file_handle)
            
            pd.testing.assert_frame_equal(result, test_data)
    
    def test_read_csv_file_not_found(self):
        """Test CSV reading when file doesn't exist in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        with pytest.raises(FileNotFoundError, match="CSV file not found in S3"):
            reader.read_csv('s3://bucket/nonexistent.csv')
    
    def test_read_excel_success(self):
        """Test successful Excel file reading from S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        
        # Mock the file handle returned by fs.open()
        mock_file_handle = MagicMock()
        mock_fs.open.return_value.__enter__.return_value = mock_file_handle
        
        test_data = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with patch('pandas.read_excel', return_value=test_data) as mock_read_excel:
            reader = S3FileReader()
            reader.fs = mock_fs
            
            result = reader.read_excel('s3://bucket/file.xlsx')
            
            # Verify that fs.open was called correctly
            mock_fs.open.assert_called_once_with('s3://bucket/file.xlsx', 'rb')
            
            # Verify that pandas.read_excel was called with the file handle
            mock_read_excel.assert_called_once_with(mock_file_handle)
            
            pd.testing.assert_frame_equal(result, test_data)
    
    def test_read_excel_file_not_found(self):
        """Test Excel reading when file doesn't exist in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        with pytest.raises(FileNotFoundError, match="Excel file not found in S3"):
            reader.read_excel('s3://bucket/nonexistent.xlsx')
    
    def test_file_exists_true(self):
        """Test file existence check when file exists in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        assert reader.file_exists('s3://bucket/file.csv') is True
        mock_fs.exists.assert_called_once_with('s3://bucket/file.csv')
    
    def test_file_exists_false(self):
        """Test file existence check when file doesn't exist in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        assert reader.file_exists('s3://bucket/nonexistent.csv') is False
    
    def test_file_exists_exception_handling(self):
        """Test file existence check with exception handling."""
        mock_fs = MagicMock()
        mock_fs.exists.side_effect = Exception("S3 error")
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        assert reader.file_exists('s3://bucket/file.csv') is False
    
    def test_get_file_size_success(self):
        """Test successful file size retrieval from S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_fs.info.return_value = {'size': 1024}
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        size = reader.get_file_size('s3://bucket/file.csv')
        
        assert size == 1024
        mock_fs.info.assert_called_once_with('s3://bucket/file.csv')
    
    def test_get_file_size_file_not_found(self):
        """Test file size retrieval when file doesn't exist in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        with pytest.raises(FileNotFoundError, match="File not found in S3"):
            reader.get_file_size('s3://bucket/nonexistent.csv')
    
    def test_get_file_size_exception_handling(self):
        """Test file size retrieval with exception handling."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_fs.info.side_effect = Exception("S3 error")
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        with pytest.raises(RuntimeError, match="Failed to get file size"):
            reader.get_file_size('s3://bucket/file.csv')


class TestS3VectorStorage:
    """Test S3VectorStorage class."""
    
    def test_init_with_kwargs(self):
        """Test S3VectorStorage initialization with kwargs."""
        with patch('cocoon.core.storage.s3.s3fs.S3FileSystem') as mock_s3fs:
            mock_fs = MagicMock()
            mock_s3fs.return_value = mock_fs
            
            storage = S3VectorStorage(profile_name='test-profile', region='us-east-1')
            
            mock_s3fs.assert_called_once_with(profile_name='test-profile', region='us-east-1')
            assert storage.fs == mock_fs
    
    def test_save_parquet_success(self):
        """Test successful parquet file saving to S3."""
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John', 'Jane']})
        
        with patch('cocoon.core.storage.s3.boto3.client') as mock_boto3:
            mock_s3_client = MagicMock()
            mock_boto3.return_value = mock_s3_client
            
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            # Mock the pandas to_parquet method to avoid actual file operations
            with patch.object(test_data, 'to_parquet') as mock_to_parquet:
                storage.save_parquet(test_data, 's3://bucket/file.parquet')
                
                # Check that bucket existence was verified
                mock_s3_client.head_bucket.assert_called_once_with(Bucket='bucket')
                
                # Check that file was opened for writing
                mock_fs.open.assert_called_once_with('s3://bucket/file.parquet', 'wb')
                
                # Check that to_parquet was called
                mock_to_parquet.assert_called_once()
    
    def test_save_parquet_with_compression(self):
        """Test parquet saving with custom compression."""
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John']})
        
        with patch('cocoon.core.storage.s3.boto3.client'):
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            # Mock the pandas to_parquet method to avoid actual file operations
            with patch.object(test_data, 'to_parquet') as mock_to_parquet:
                storage.save_parquet(test_data, 's3://bucket/file.parquet', compression='gzip')
                
                mock_fs.open.assert_called_once_with('s3://bucket/file.parquet', 'wb')
                mock_to_parquet.assert_called_once()
    
    def test_save_csv_success(self):
        """Test successful CSV file saving to S3."""
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John', 'Jane']})
        
        with patch('cocoon.core.storage.s3.boto3.client'):
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            storage.save_csv(test_data, 's3://bucket/file.csv')
            
            mock_fs.open.assert_called_once_with('s3://bucket/file.csv', 'w')
    
    def test_save_csv_with_encoding(self):
        """Test CSV saving with custom encoding."""
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John']})
        
        with patch('cocoon.core.storage.s3.boto3.client'):
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            storage.save_csv(test_data, 's3://bucket/file.csv', encoding='latin-1')
            
            mock_fs.open.assert_called_once_with('s3://bucket/file.csv', 'w')
    
    def test_save_jsonl_success(self):
        """Test successful JSONL file saving to S3."""
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John', 'Jane']})
        
        with patch('cocoon.core.storage.s3.boto3.client'):
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            storage.save_jsonl(test_data, 's3://bucket/file.jsonl')
            
            mock_fs.open.assert_called_once_with('s3://bucket/file.jsonl', 'w')
    
    def test_save_jsonl_with_custom_params(self):
        """Test JSONL saving with custom parameters."""
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John']})
        
        with patch('cocoon.core.storage.s3.boto3.client'):
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            # Mock the pandas to_json method to avoid actual file operations
            with patch.object(test_data, 'to_json') as mock_to_json:
                storage.save_jsonl(test_data, 's3://bucket/file.jsonl', orient='records')
                
                mock_fs.open.assert_called_once_with('s3://bucket/file.jsonl', 'w')
                mock_to_json.assert_called_once()
    
    def test_directory_exists_true(self):
        """Test directory existence check when directory exists in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        assert storage.directory_exists('s3://bucket/directory/') is True
        mock_fs.exists.assert_called_once_with('s3://bucket/directory/')
    
    def test_directory_exists_false(self):
        """Test directory existence check when directory doesn't exist in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        assert storage.directory_exists('s3://bucket/nonexistent/') is False
    
    def test_directory_exists_exception_handling(self):
        """Test directory existence check with exception handling."""
        mock_fs = MagicMock()
        mock_fs.exists.side_effect = Exception("S3 error")
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        assert storage.directory_exists('s3://bucket/directory/') is False
    
    def test_create_directory_success(self):
        """Test successful directory creation in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        storage.create_directory('s3://bucket/new_directory/')
        
        mock_fs.touch.assert_called_once_with('s3://bucket/new_directory/')
    
    def test_create_directory_already_exists(self):
        """Test directory creation when directory already exists in S3."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        storage.create_directory('s3://bucket/existing_directory/')
        
        mock_fs.touch.assert_not_called()
    
    def test_create_directory_exception_handling(self):
        """Test directory creation with exception handling."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        mock_fs.touch.side_effect = Exception("S3 error")
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        # Should not raise an exception, just print a warning
        storage.create_directory('s3://bucket/directory/')
    
    def test_ensure_bucket_exists_success(self):
        """Test successful bucket existence verification."""
        mock_fs = MagicMock()
        
        with patch('cocoon.core.storage.s3.boto3.client') as mock_boto3:
            mock_s3_client = MagicMock()
            mock_boto3.return_value = mock_s3_client
            
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            # Should not raise an exception
            storage._ensure_bucket_exists('s3://bucket/file.csv')
            
            mock_s3_client.head_bucket.assert_called_once_with(Bucket='bucket')
    
    def test_ensure_bucket_exists_failure(self):
        """Test bucket existence verification failure."""
        mock_fs = MagicMock()
        
        with patch('cocoon.core.storage.s3.boto3.client') as mock_boto3:
            mock_s3_client = MagicMock()
            mock_s3_client.head_bucket.side_effect = Exception("Bucket not found")
            mock_boto3.return_value = mock_s3_client
            
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            with pytest.raises(RuntimeError, match="S3 bucket bucket does not exist"):
                storage._ensure_bucket_exists('s3://bucket/file.csv')
    
    def test_ensure_bucket_exists_invalid_path(self):
        """Test bucket existence verification with invalid S3 path."""
        mock_fs = MagicMock()
        
        storage = S3VectorStorage()
        storage.fs = mock_fs
        
        # Should not raise an exception for non-S3 paths
        storage._ensure_bucket_exists('local/file.csv')


class TestS3StorageIntegration:
    """Test S3 storage integration scenarios."""
    
    def test_s3_path_handling(self):
        """Test S3 path handling in various scenarios."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = True
        mock_fs.open.return_value.__enter__ = MagicMock()
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=None)
        
        test_data = pd.DataFrame({'name': ['John']})
        
        with patch('cocoon.core.storage.s3.boto3.client'):
            storage = S3VectorStorage()
            storage.fs = mock_fs
            
            # Test various S3 path formats
            paths = [
                's3://bucket/file.parquet',
                's3://my-bucket/data/file.csv',
                's3://deep/nested/path/file.jsonl'
            ]
            
            for path in paths:
                # Mock the pandas to_parquet method to avoid actual file operations
                with patch.object(test_data, 'to_parquet'):
                    storage.save_parquet(test_data, path)
            
            # Verify that open was called for each path
            assert mock_fs.open.call_count == 3
            mock_fs.open.assert_any_call('s3://bucket/file.parquet', 'wb')
            mock_fs.open.assert_any_call('s3://my-bucket/data/file.csv', 'wb')
            mock_fs.open.assert_any_call('s3://deep/nested/path/file.jsonl', 'wb')
    
    def test_error_handling_integration(self):
        """Test error handling integration in S3 operations."""
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        
        reader = S3FileReader()
        reader.fs = mock_fs
        
        # Test file not found errors
        with pytest.raises(FileNotFoundError):
            reader.read_csv('s3://bucket/nonexistent.csv')
        
        with pytest.raises(FileNotFoundError):
            reader.read_excel('s3://bucket/nonexistent.xlsx')
        
        with pytest.raises(FileNotFoundError):
            reader.get_file_size('s3://bucket/nonexistent.txt')
