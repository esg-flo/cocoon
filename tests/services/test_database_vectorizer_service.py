"""Unit tests for the DatabaseVectorizerService class."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from cocoon.services.database_vectorizer import DatabaseVectorizerService
from cocoon.core.config.models import (
    VectorizationPipelineConfig, FileInputConfig, ProcessingConfig,
    VectorizationConfig, OutputConfig
)


class TestDatabaseVectorizerService:
    """Test cases for DatabaseVectorizerService."""
    
    def create_test_config(self) -> VectorizationPipelineConfig:
        """Create a test configuration."""
        return VectorizationPipelineConfig(
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
                embedding_model_config={
                    "model_id": "amazon.titan-embed-text-v1",
                    "aws_region_name": "us-east-1"
                },
                target_column="text",
                metadata_columns=["id", "category"],
                batch_size=100
            ),
            output=OutputConfig(
                output_path="output.parquet",
                output_format="parquet",
                compression="snappy"
            )
        )
    
    def test_init_with_config(self):
        """Test initialization with a provided configuration."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        assert service.config == config
        assert service.logger is not None
    
    def test_init_without_config(self):
        """Test initialization without configuration (uses defaults)."""
        with patch('cocoon.services.database_vectorizer.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'DB_VECTORIZER_DEFAULT_EMBEDDING_MODEL': 'amazon.titan-embed-text-v1',
                'AWS_DEFAULT_REGION': 'us-east-1',
                'DB_VECTORIZER_DEFAULT_BATCH_SIZE': '100'
            }.get(key, default)
            
            service = DatabaseVectorizerService()
            
            assert service.config is not None
            assert service.config.input.file_path == "placeholder.csv"
            assert service.config.vectorization.target_column == "text"
            assert service.config.output.output_path == "output.parquet"
    
    def test_process_file_success(self):
        """Test successful file processing."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        # Mock the DatabaseVectorizer
        with patch('cocoon.services.database_vectorizer.DatabaseVectorizer') as mock_vectorizer_class:
            mock_vectorizer = MagicMock()
            mock_result_df = pd.DataFrame({
                'label': ['Hello World', 'Test Text'],
                'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                'index_ids': [[0], [1]],
                'id': [1, 2],
                'category': ['A', 'B']
            })
            mock_vectorizer.process.return_value = mock_result_df
            mock_vectorizer.get_statistics.return_value = {
                'input_file': 'test.csv',
                'output_file': 'output.parquet',
                'target_column': 'text'
            }
            mock_vectorizer_class.return_value = mock_vectorizer
            
            # Process file
            result = service.process_file(
                input_file_path="test.csv",
                target_column="text",
                output_path="output.parquet",
                metadata_columns=["id", "category"]
            )
            
            # Verify results
            assert result['status'] == 'success'
            assert result['rows_processed'] == 2
            assert result['unique_texts'] == 2
            assert 'input_file' in result
    
    def test_process_file_with_kwargs(self):
        """Test file processing with additional configuration overrides."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        # Mock the DatabaseVectorizer
        with patch('cocoon.services.database_vectorizer.DatabaseVectorizer') as mock_vectorizer_class:
            mock_vectorizer = MagicMock()
            mock_result_df = pd.DataFrame({'label': ['Test'], 'embedding': [[0.1, 0.2, 0.3]]})
            mock_vectorizer.process.return_value = mock_result_df
            mock_vectorizer.get_statistics.return_value = {}
            mock_vectorizer_class.return_value = mock_vectorizer
            
            # Process file with overrides
            result = service.process_file(
                input_file_path="test.csv",
                target_column="text",
                output_path="output.csv",
                output_format="csv",
                deduplicate_text=False,
                batch_size=50
            )
            
            # Verify the result was successful
            assert result['status'] == 'success'
    
    def test_process_file_error_handling(self):
        """Test error handling during file processing."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        # Mock the DatabaseVectorizer to raise an error
        with patch('cocoon.services.database_vectorizer.DatabaseVectorizer') as mock_vectorizer_class:
            mock_vectorizer = MagicMock()
            mock_vectorizer.process.side_effect = ValueError("Test error")
            mock_vectorizer_class.return_value = mock_vectorizer
            
            # Process file and expect error
            result = service.process_file(
                input_file_path="test.csv",
                target_column="text",
                output_path="output.parquet"
            )
            
            # Verify error statistics were recorded
            assert result['status'] == 'error'
            assert result['error_message'] == "Test error"
            assert result['error_type'] == "ValueError"
    
    def test_create_file_config(self):
        """Test configuration creation for file processing."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        # Create configuration for a file
        file_config = service._create_file_config(
            input_file_path="new.csv",
            target_column="description",
            output_path="new_output.parquet",
            metadata_columns=["id", "price"],
            output_format="jsonl"
        )
        
        # Verify updates
        assert file_config.input.file_path == "new.csv"
        assert file_config.vectorization.target_column == "description"
        assert file_config.output.output_path == "new_output.parquet"
        assert file_config.vectorization.metadata_columns == ["id", "price"]
        assert file_config.output.output_format == "jsonl"
        
        # Verify original config is unchanged
        assert service.config.input.file_path == "test.csv"
    
    def test_health_check_success(self):
        """Test successful health check."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        with patch('cocoon.services.database_vectorizer.BedrockEmbeddings') as mock_bedrock:
            mock_instance = MagicMock()
            mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_bedrock.return_value = mock_instance
            
            health = service.health_check()
            
            assert health['status'] == 'healthy'
            assert health['embeddings_available'] is True
            assert health['test_embedding_dimensions'] == 3
            assert health['configuration_valid'] is True
            assert health['config_loaded'] is True
    
    def test_health_check_failure(self):
        """Test health check failure."""
        config = self.create_test_config()
        service = DatabaseVectorizerService(config)
        
        with patch('cocoon.services.database_vectorizer.BedrockEmbeddings') as mock_bedrock:
            mock_bedrock.side_effect = Exception("Connection failed")
            
            health = service.health_check()
            
            assert health['status'] == 'unhealthy'
            assert health['embeddings_available'] is False
            assert 'error' in health
            assert health['configuration_valid'] is False
            assert health['config_loaded'] is True


class TestDatabaseVectorizerServiceIntegration:
    """Integration tests for DatabaseVectorizerService."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing with mocked components."""
        from cocoon.services.database_vectorizer import DatabaseVectorizerService
        
        # Create service with test config
        service = DatabaseVectorizerService()
        
        # Mock the underlying vectorizer
        with patch('cocoon.services.database_vectorizer.DatabaseVectorizer') as mock_vectorizer_class:
            mock_vectorizer = MagicMock()
            mock_result_df = pd.DataFrame({
                'label': ['Hello World', 'Test Text'],
                'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                'index_ids': [[0], [1]],
                'id': [1, 2],
                'category': ['A', 'B']
            })
            mock_vectorizer.process.return_value = mock_result_df
            mock_vectorizer.get_statistics.return_value = {
                'input_file': 'test.csv',
                'output_file': 'output.parquet',
                'target_column': 'text'
            }
            mock_vectorizer_class.return_value = mock_vectorizer
            
            # Process a file
            result = service.process_file(
                input_file_path="test.csv",
                target_column="text",
                output_path="output.parquet",
                metadata_columns=["id", "category"]
            )
            
            # Verify the complete flow
            assert result['status'] == 'success'
            assert result['rows_processed'] == 2
            assert result['unique_texts'] == 2
    
    def test_multiple_file_processing(self):
        """Test that processing multiple files doesn't interfere with each other."""
        service = DatabaseVectorizerService()
        
        # Mock the DatabaseVectorizer
        with patch('cocoon.services.database_vectorizer.DatabaseVectorizer') as mock_vectorizer_class:
            mock_vectorizer = MagicMock()
            mock_result_df = pd.DataFrame({'label': ['Test'], 'embedding': [[0.1, 0.2, 0.3]]})
            mock_vectorizer.process.return_value = mock_result_df
            mock_vectorizer.get_statistics.return_value = {}
            mock_vectorizer_class.return_value = mock_vectorizer
            
            # Process first file
            result1 = service.process_file(
                input_file_path="file1.csv",
                target_column="text1",
                output_path="output1.parquet"
            )
            
            # Process second file
            result2 = service.process_file(
                input_file_path="file2.csv",
                target_column="text2",
                output_path="output2.parquet"
            )
            
            # Both should succeed independently
            assert result1['status'] == 'success'
            assert result2['status'] == 'success'
