"""Unit tests for local storage implementation."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from cocoon.core.storage.local import LocalFileReader, LocalVectorStorage


class TestLocalFileReader:
    """Test LocalFileReader class."""
    
    def test_read_csv_success(self):
        """Test successful CSV file reading."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age\nJohn,30\nJane,25")
            temp_file = f.name
        
        try:
            reader = LocalFileReader()
            df = reader.read_csv(temp_file)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['name', 'age']
            assert df.iloc[0]['name'] == 'John'
            assert df.iloc[0]['age'] == 30
        finally:
            os.unlink(temp_file)
    
    def test_read_csv_with_kwargs(self):
        """Test CSV reading with additional kwargs."""
        # Create a temporary CSV file with custom delimiter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name;age\nJohn;30\nJane;25")
            temp_file = f.name
        
        try:
            reader = LocalFileReader()
            df = reader.read_csv(temp_file, delimiter=';')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['name', 'age']
        finally:
            os.unlink(temp_file)
    
    def test_read_csv_file_not_found(self):
        """Test CSV reading when file doesn't exist."""
        reader = LocalFileReader()
        
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            reader.read_csv("nonexistent_file.csv")
    
    def test_read_excel_success(self):
        """Test successful Excel file reading."""
        # Create a temporary Excel file
        df_data = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_file = f.name
        
        try:
            df_data.to_excel(temp_file, index=False)
            
            reader = LocalFileReader()
            df = reader.read_excel(temp_file)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['name', 'age']
        finally:
            os.unlink(temp_file)
    
    def test_read_excel_file_not_found(self):
        """Test Excel reading when file doesn't exist."""
        reader = LocalFileReader()
        
        with pytest.raises(FileNotFoundError, match="Excel file not found"):
            reader.read_excel("nonexistent_file.xlsx")
    
    def test_file_exists_true(self):
        """Test file existence check when file exists."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            reader = LocalFileReader()
            assert reader.file_exists(temp_file) is True
        finally:
            os.unlink(temp_file)
    
    def test_file_exists_false(self):
        """Test file existence check when file doesn't exist."""
        reader = LocalFileReader()
        assert reader.file_exists("nonexistent_file.txt") is False
    
    def test_get_file_size_success(self):
        """Test successful file size retrieval."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            reader = LocalFileReader()
            size = reader.get_file_size(temp_file)
            
            assert size > 0
            assert size == len("test content")
        finally:
            os.unlink(temp_file)
    
    def test_get_file_size_file_not_found(self):
        """Test file size retrieval when file doesn't exist."""
        reader = LocalFileReader()
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            reader.get_file_size("nonexistent_file.txt")


class TestLocalVectorStorage:
    """Test LocalVectorStorage class."""
    
    def test_save_parquet_success(self):
        """Test successful parquet file saving."""
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            output_path = Path(temp_dir) / "test.parquet"
            
            storage.save_parquet(df, output_path)
            
            assert output_path.exists()
            
            # Verify the saved data
            saved_df = pd.read_parquet(output_path)
            pd.testing.assert_frame_equal(df, saved_df)
    
    def test_save_parquet_with_compression(self):
        """Test parquet saving with custom compression."""
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            output_path = Path(temp_dir) / "test.parquet"
            
            storage.save_parquet(df, output_path, compression='gzip')
            
            assert output_path.exists()
    
    def test_save_csv_success(self):
        """Test successful CSV file saving."""
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            output_path = Path(temp_dir) / "test.csv"
            
            storage.save_csv(df, output_path)
            
            assert output_path.exists()
            
            # Verify the saved data
            saved_df = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(df, saved_df)
    
    def test_save_csv_with_encoding(self):
        """Test CSV saving with custom encoding."""
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            output_path = Path(temp_dir) / "test.csv"
            
            storage.save_csv(df, output_path, encoding='latin-1')
            
            assert output_path.exists()
    
    def test_save_jsonl_success(self):
        """Test successful JSONL file saving."""
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            output_path = Path(temp_dir) / "test.jsonl"
            
            storage.save_jsonl(df, output_path)
            
            assert output_path.exists()
            
            # Verify the saved data
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert '"name":"John"' in lines[0]
                assert '"name":"Jane"' in lines[1]
    
    def test_save_jsonl_with_custom_params(self):
        """Test JSONL saving with custom parameters."""
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            output_path = Path(temp_dir) / "test.jsonl"
            
            # Test with valid JSONL parameters
            storage.save_jsonl(df, output_path, orient='records')
            
            assert output_path.exists()
    
    def test_directory_exists_true(self):
        """Test directory existence check when directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            assert storage.directory_exists(temp_dir) is True
    
    def test_directory_exists_false(self):
        """Test directory existence check when directory doesn't exist."""
        storage = LocalVectorStorage()
        assert storage.directory_exists("nonexistent_directory") is False
    
    def test_create_directory_success(self):
        """Test successful directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            new_dir = Path(temp_dir) / "new_subdirectory"
            
            storage.create_directory(new_dir)
            
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_create_directory_already_exists(self):
        """Test directory creation when directory already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            existing_dir = Path(temp_dir)
            
            # Should not raise an error
            storage.create_directory(existing_dir)
            
            assert existing_dir.exists()
    
    def test_ensure_directory_exists_creates_parent(self):
        """Test that parent directories are created when saving files."""
        df = pd.DataFrame({'name': ['John']})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalVectorStorage()
            nested_path = Path(temp_dir) / "level1" / "level2" / "test.parquet"
            
            storage.save_parquet(df, nested_path)
            
            assert nested_path.exists()
            assert nested_path.parent.exists()
            assert (nested_path.parent.parent).exists()


class TestLocalStorageIntegration:
    """Test local storage integration scenarios."""
    
    def test_read_and_save_roundtrip(self):
        """Test reading a file and saving it back."""
        # Create test data
        original_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original data as CSV
            storage = LocalVectorStorage()
            original_path = Path(temp_dir) / "original.csv"
            storage.save_csv(original_df, original_path)
            
            # Read it back
            reader = LocalFileReader()
            read_df = reader.read_csv(original_path)
            
            # Verify data integrity
            pd.testing.assert_frame_equal(original_df, read_df)
            
            # Save in different format
            parquet_path = Path(temp_dir) / "converted.parquet"
            storage.save_parquet(original_df, parquet_path)
            
            # Verify both files exist
            assert original_path.exists()
            assert parquet_path.exists()
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        storage = LocalVectorStorage()
        reader = LocalFileReader()
        
        # Test file not found errors
        with pytest.raises(FileNotFoundError):
            reader.read_csv("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            reader.read_excel("nonexistent.xlsx")
        
        with pytest.raises(FileNotFoundError):
            reader.get_file_size("nonexistent.txt")
