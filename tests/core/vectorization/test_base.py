"""Unit tests for the core DatabaseVectorizer class."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import numpy as np

from cocoon.core.vectorization.base import DatabaseVectorizer
from cocoon.core.config.models import (
    FileInputConfig,
    ProcessingConfig,
    VectorizationConfig,
    OutputConfig,
    VectorizationPipelineConfig
)


class TestDatabaseVectorizer:
    """Test DatabaseVectorizer class."""
    
    def create_test_config(self, file_path: str = "test.csv", output_path: str = "output.parquet") -> VectorizationPipelineConfig:
        """Create a test configuration."""
        input_config = FileInputConfig(
            file_path=file_path,
            file_type="csv",
            csv_delimiter=",",
            csv_encoding="utf-8"
        )
        
        processing_config = ProcessingConfig(
            deduplicate_text=True,
            preserve_original_indices=True,
            text_cleaning=True
        )
        
        vectorization_config = VectorizationConfig(
            embedding_model_config={
                "model_id": "amazon.titan-embed-text-v1",
                "aws_region_name": "eu-central-1"
            },
            target_column="text",
            metadata_columns=["id", "category"],
            batch_size=2
        )
        
        output_config = OutputConfig(
            output_path=output_path,
            output_format="parquet",
            compression="snappy",
            include_metadata=True
        )
        
        return VectorizationPipelineConfig(
            input=input_config,
            processing=processing_config,
            vectorization=vectorization_config,
            output=output_config
        )
    
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            assert vectorizer.config == config
            assert vectorizer.file_reader is not None
            assert vectorizer.vector_storage is not None
            assert vectorizer.embeddings is not None
    
    def test_init_with_s3_config(self):
        """Test initialization with S3 configuration."""
        config = self.create_test_config("s3://bucket/file.csv", "s3://bucket/output.parquet")
        
        with patch('cocoon.core.vectorization.base.S3FileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.S3VectorStorage') as mock_storage:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            assert vectorizer.config == config
            # Should use S3 components
            mock_reader.assert_called_once()
            mock_storage.assert_called_once()
    
    def test_create_file_reader_local(self):
        """Test file reader creation for local files."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader:
            mock_reader.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            mock_reader.assert_called_once()
    
    def test_create_file_reader_s3(self):
        """Test file reader creation for S3 files."""
        config = self.create_test_config("s3://bucket/file.csv")
        
        with patch('cocoon.core.vectorization.base.S3FileReader') as mock_reader:
            mock_reader.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            mock_reader.assert_called_once()
    
    def test_create_vector_storage_local(self):
        """Test vector storage creation for local output."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage:
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            mock_storage.assert_called_once()
    
    def test_create_vector_storage_s3(self):
        """Test vector storage creation for S3 output."""
        config = self.create_test_config(output_path="s3://bucket/output.parquet")
        
        with patch('cocoon.core.vectorization.base.S3VectorStorage') as mock_storage:
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            mock_storage.assert_called_once()
    
    def test_create_embeddings_with_config(self):
        """Test embeddings creation with custom configuration."""
        config = self.create_test_config()
        config.vectorization.embedding_model_config = {
            'model_id': 'custom-model',
            'aws_region_name': 'us-west-2'
        }
        
        vectorizer = DatabaseVectorizer(config)
        
        # Verify that embeddings were created (the mock fallback will be used)
        assert vectorizer.embeddings is not None
    
    def test_create_embeddings_default(self):
        """Test embeddings creation with default configuration."""
        config = self.create_test_config()
        config.vectorization.embedding_model_config = {}
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_bedrock:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_bedrock.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            # Verify that embeddings were created
            assert vectorizer.embeddings is not None
    
    def test_read_input_file_csv(self):
        """Test reading CSV input file."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            # Mock the file reader
            test_df = pd.DataFrame({
                'text': ['Hello World', 'Test Text'],
                'id': [1, 2],
                'category': ['A', 'B']
            })
            vectorizer.file_reader.read_csv.return_value = test_df
            
            result = vectorizer._read_input_file()
            
            pd.testing.assert_frame_equal(result, test_df)
            vectorizer.file_reader.read_csv.assert_called_once_with(
                "test.csv",
                delimiter=",",
                encoding="utf-8"
            )
    
    def test_read_input_file_excel(self):
        """Test reading Excel input file."""
        config = self.create_test_config("test.xlsx")
        config.input.file_type = "excel"
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            # Mock the file reader
            test_df = pd.DataFrame({
                'text': ['Hello World', 'Test Text'],
                'id': [1, 2],
                'category': ['A', 'B']
            })
            vectorizer.file_reader.read_excel.return_value = test_df
            
            result = vectorizer._read_input_file()
            
            pd.testing.assert_frame_equal(result, test_df)
            vectorizer.file_reader.read_excel.assert_called_once_with(
                "test.xlsx",
                sheet_name=None,
                engine="openpyxl"
            )
    
    def test_read_input_file_unsupported_type(self):
        """Test reading unsupported file type."""
        config = self.create_test_config("test.txt")
        config.input.file_type = "txt"
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            with pytest.raises(ValueError, match="Unsupported file type: txt"):
                vectorizer._read_input_file()
    
    def test_process_text_data_basic(self):
        """Test basic text processing."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            input_df = pd.DataFrame({
                'text': ['  Hello World  ', '  Test Text  ', '  Hello World  '],
                'id': [1, 2, 3],
                'category': ['A', 'B', 'A']
            })
            
            result = vectorizer._process_text_data(input_df)
            
            # Should deduplicate and clean text
            assert len(result) == 2  # Duplicate removed
            assert 'label' in result.columns
            assert 'index_ids' in result.columns
            # Label column should preserve original text (not cleaned)
            assert result.iloc[0]['label'] == '  Hello World  '
            assert result.iloc[1]['label'] == '  Test Text  '
    
    def test_process_text_data_missing_target_column(self):
        """Test text processing with missing target column."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            input_df = pd.DataFrame({
                'wrong_column': ['Hello World', 'Test Text'],
                'id': [1, 2]
            })
            
            with pytest.raises(ValueError, match="Target column 'text' not found in input file"):
                vectorizer._process_text_data(input_df)
    
    def test_process_text_data_no_deduplication(self):
        """Test text processing without deduplication."""
        config = self.create_test_config()
        config.processing.deduplicate_text = False
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            input_df = pd.DataFrame({
                'text': ['Hello World', 'Test Text', 'Hello World'],
                'id': [1, 2, 3],
                'category': ['A', 'B', 'A']
            })
            
            result = vectorizer._process_text_data(input_df)
            
            # Should not deduplicate
            assert len(result) == 3  # All rows preserved
            assert 'label' in result.columns
            # When no deduplication, index_ids is not created
            assert 'index_ids' not in result.columns
            # Label column should preserve original text
            assert result.iloc[0]['label'] == 'Hello World'
            assert result.iloc[1]['label'] == 'Test Text'
            assert result.iloc[2]['label'] == 'Hello World'
    
    def test_process_text_data_text_cleaning(self):
        """Test text processing with text cleaning."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            input_df = pd.DataFrame({
                'text': ['  Hello   World  ', '  Test   Text  ', '  Another   Text  '],
                'id': [1, 2, 3],
                'category': ['A', 'B', 'C']
            })
            
            result = vectorizer._process_text_data(input_df)
            
            # Should deduplicate and clean text
            assert len(result) == 3  # All unique texts
            assert 'label' in result.columns
            assert 'index_ids' in result.columns
            # Label column should preserve original text (not cleaned)
            # Check that all expected labels are present (order may vary due to deduplication)
            expected_labels = ['  Hello   World  ', '  Test   Text  ', '  Another   Text  ']
            actual_labels = result['label'].tolist()
            for expected_label in expected_labels:
                assert expected_label in actual_labels
    
    def test_generate_embeddings(self):
        """Test embedding generation."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            # Mock embeddings
            mock_embeddings_instance = MagicMock()
            mock_embeddings_instance.embed_documents.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
            vectorizer.embeddings = mock_embeddings_instance
            
            # Test with DataFrame that has the structure from _process_text_data
            input_df = pd.DataFrame({
                'text': ['Hello World', 'Test Text', 'Hello World'],  # Original data with duplicates
                'id': [1, 2, 3],
                'category': ['A', 'B', 'A']
            })
            
            # Process the text data first (this creates label and index_ids)
            processed_df = vectorizer._process_text_data(input_df)
            
            # Now generate embeddings
            result = vectorizer._generate_embeddings(processed_df)
            
            # Should have embeddings column
            assert 'embedding' in result.columns
            assert len(result['embedding']) == 2  # Only 2 unique texts
            assert result.iloc[0]['embedding'] == [0.1, 0.2, 0.3]
            assert result.iloc[1]['embedding'] == [0.4, 0.5, 0.6]
            
            # Check that required columns are present in the expected order
            assert result.columns[0] == 'label'
            assert result.columns[1] == 'embedding'
            assert result.columns[2] == 'index_ids'
            # Metadata columns should be present but order may vary
            assert 'id' in result.columns
            assert 'category' in result.columns
    
    def test_save_output_parquet(self):
        """Test saving output as parquet."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            test_df = pd.DataFrame({
                'label': ['Hello World'],
                'embedding': [[0.1, 0.2, 0.3]],
                'index_ids': [[0]],
                'id': [1],
                'category': ['A']
            })
            
            vectorizer._save_output(test_df)
            
            vectorizer.vector_storage.save_parquet.assert_called_once_with(
                test_df,
                "output.parquet",
                compression="snappy"
            )
    
    def test_save_output_csv(self):
        """Test saving output as CSV."""
        config = self.create_test_config()
        config.output.output_format = "csv"
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            test_df = pd.DataFrame({
                'label': ['Hello World'],
                'embedding': [[0.1, 0.2, 0.3]],
                'index_ids': [[0]],
                'id': [1],
                'category': ['A']
            })
            
            vectorizer._save_output(test_df)
            
            vectorizer.vector_storage.save_csv.assert_called_once_with(
                test_df,
                "output.parquet"  # Still uses original path, but no compression for CSV
            )
    
    def test_save_output_jsonl(self):
        """Test saving output as JSONL."""
        config = self.create_test_config()
        config.output.output_format = "jsonl"
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            test_df = pd.DataFrame({
                'label': ['Hello World'],
                'embedding': [[0.1, 0.2, 0.3]],
                'index_ids': [[0]],
                'id': [1],
                'category': ['A']
            })
            
            vectorizer._save_output(test_df)
            
            vectorizer.vector_storage.save_jsonl.assert_called_once_with(
                test_df,
                "output.parquet"  # Still uses original path
            )
    
    def test_save_output_unsupported_format(self):
        """Test saving output with unsupported format."""
        config = self.create_test_config()
        config.output.output_format = "unsupported"
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            test_df = pd.DataFrame({
                'label': ['Hello World'],
                'embedding': [[0.1, 0.2, 0.3]],
                'index_ids': [[0]]
            })
            
            with pytest.raises(ValueError, match="Unsupported output format: unsupported"):
                vectorizer._save_output(test_df)
    
    def test_get_statistics(self):
        """Test getting processing statistics."""
        config = self.create_test_config()
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            mock_reader.return_value = MagicMock()
            mock_storage.return_value = MagicMock()
            mock_embeddings.return_value = MagicMock()
            
            vectorizer = DatabaseVectorizer(config)
            
            stats = vectorizer.get_statistics()
            
            expected_keys = [
                'input_file', 'output_file', 'input_format', 'output_format',
                'target_column', 'metadata_columns', 'batch_size',
                'deduplication_enabled', 'text_cleaning_enabled'
            ]
            
            for key in expected_keys:
                assert key in stats
            
            assert stats['input_file'] == "test.csv"
            assert stats['output_file'] == "output.parquet"
            assert stats['target_column'] == "text"
            assert stats['deduplication_enabled'] is True
            assert stats['text_cleaning_enabled'] is True


class TestDatabaseVectorizerIntegration:
    """Test DatabaseVectorizer integration scenarios."""
    
    def test_complete_processing_pipeline(self):
        """Test the complete processing pipeline."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(
                file_path="test.csv",
                file_type="csv",
                csv_delimiter=",",
                csv_encoding="utf-8"
            ),
            processing=ProcessingConfig(
                deduplicate_text=True,
                preserve_original_indices=True,
                text_cleaning=True
            ),
            vectorization=VectorizationConfig(
                embedding_model_config={},
                target_column="text",
                metadata_columns=["id"],
                batch_size=2
            ),
            output=OutputConfig(
                output_path="output.parquet",
                output_format="parquet"
            )
        )
        
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage') as mock_storage, \
             patch('cocoon.core.embeddings.bedrock.BedrockEmbeddings') as mock_embeddings:
            
            # Mock file reader
            mock_reader_instance = MagicMock()
            test_df = pd.DataFrame({
                'text': ['  Hello World  ', '  Test Text  ', '  Hello World  '],
                'id': [1, 2, 3]
            })
            mock_reader_instance.read_csv.return_value = test_df
            mock_reader.return_value = mock_reader_instance
            
            # Mock storage
            mock_storage_instance = MagicMock()
            mock_storage.return_value = mock_storage_instance
            
            # Mock embeddings
            mock_embeddings_instance = MagicMock()
            mock_embeddings_instance.embed_documents.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
            mock_embeddings.return_value = mock_embeddings_instance
            
            vectorizer = DatabaseVectorizer(config)
            
            # Run the complete pipeline
            result = vectorizer.process()
            
            # Verify the result
            assert len(result) == 2  # Duplicate removed
            assert 'label' in result.columns
            assert 'embedding' in result.columns
            assert 'index_ids' in result.columns
            assert 'id' in result.columns
            
            # Verify file operations were called
            mock_reader_instance.read_csv.assert_called_once()
            mock_storage_instance.save_parquet.assert_called_once()
            mock_embeddings_instance.embed_documents.assert_called()
            
            # Verify logging calls
            # Note: We can't easily test logging without more complex mocking
