"""Test integration between TextProcessor and DatabaseVectorizer."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from cocoon.core.vectorization.base import DatabaseVectorizer
from cocoon.core.vectorization.processors import TextProcessor
from cocoon.core.config.models import (
    VectorizationPipelineConfig,
    FileInputConfig,
    ProcessingConfig,
    VectorizationConfig,
    OutputConfig
)


class TestTextProcessingIntegration:
    """Test the integration between TextProcessor and DatabaseVectorizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'product_id': [1, 2, 3],
            'product_description': [
                'High-Quality Laptop (2023 Model) - 16GB RAM',
                'Gaming Mouse with RGB Lighting & 25K DPI',
                'Wireless Headphones - Noise Cancelling + Bluetooth 5.0'
            ],
            'category': ['Electronics', 'Gaming', 'Audio']
        })
    
    @pytest.fixture
    def basic_config(self):
        """Create basic configuration."""
        return VectorizationPipelineConfig(
            input=FileInputConfig(file_path="test.csv"),
            processing=ProcessingConfig(
                deduplicate_text=False,  # Disable for testing
                preserve_original_indices=True,
                text_cleaning=True,
                text_cleaning_options={
                    "lowercase": True,
                    "remove_punctuation": False,
                    "remove_numbers": False,
                    "remove_special_chars": False,
                    "normalize_whitespace": True
                }
            ),
            vectorization=VectorizationConfig(
                target_column="product_description",
                metadata_columns=["product_id", "category"],
                batch_size=100
            ),
            output=OutputConfig(
                output_path="test_output.parquet",
                output_format="parquet"
            )
        )
    
    @pytest.fixture
    def aggressive_config(self):
        """Create aggressive text cleaning configuration."""
        return VectorizationPipelineConfig(
            input=FileInputConfig(file_path="test.csv"),
            processing=ProcessingConfig(
                deduplicate_text=False,  # Disable for testing
                preserve_original_indices=True,
                text_cleaning=True,
                text_cleaning_options={
                    "lowercase": True,
                    "remove_punctuation": True,
                    "remove_numbers": True,
                    "remove_special_chars": True,
                    "normalize_whitespace": True
                }
            ),
            vectorization=VectorizationConfig(
                target_column="product_description",
                metadata_columns=["product_id", "category"],
                batch_size=100
            ),
            output=OutputConfig(
                output_path="test_output.parquet",
                output_format="parquet"
            )
        )
    
    def test_text_processor_initialization(self, basic_config):
        """Test that TextProcessor is properly initialized."""
        with patch('cocoon.core.vectorization.base.LocalFileReader'), \
             patch('cocoon.core.vectorization.base.LocalVectorStorage'), \
             patch.object(DatabaseVectorizer, '_create_embeddings', return_value=Mock()):
            
            vectorizer = DatabaseVectorizer(basic_config)
            assert hasattr(vectorizer, 'text_processor')
            assert isinstance(vectorizer.text_processor, TextProcessor)
    
    def test_basic_text_cleaning(self, basic_config, sample_data):
        """Test basic text cleaning with default options."""
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage'), \
             patch.object(DatabaseVectorizer, '_create_embeddings', return_value=Mock()):
            
            # Mock the file reader to return our sample data
            mock_reader.return_value.read_csv.return_value = sample_data
            
            vectorizer = DatabaseVectorizer(basic_config)
            
            # Test the text cleaning directly
            cleaned_text = vectorizer.text_processor.clean_text_with_config(
                "High-Quality Laptop (2023 Model) - 16GB RAM",
                basic_config.processing.text_cleaning_options
            )
            
            # Should lowercase and normalize whitespace, but keep punctuation and numbers
            expected = "high-quality laptop (2023 model) - 16gb ram"
            assert cleaned_text == expected
    
    def test_aggressive_text_cleaning(self, aggressive_config, sample_data):
        """Test aggressive text cleaning that removes punctuation and numbers."""
        with patch('cocoon.core.vectorization.base.LocalFileReader') as mock_reader, \
             patch('cocoon.core.vectorization.base.LocalVectorStorage'), \
             patch.object(DatabaseVectorizer, '_create_embeddings', return_value=Mock()):
            
            # Mock the file reader to return our sample data
            mock_reader.return_value.read_csv.return_value = sample_data
            
            vectorizer = DatabaseVectorizer(aggressive_config)
            
            # Test the text cleaning directly
            cleaned_text = vectorizer.text_processor.clean_text_with_config(
                "High-Quality Laptop (2023 Model) - 16GB RAM",
                aggressive_config.processing.text_cleaning_options
            )
            
            # Should remove punctuation, numbers, and special characters
            # Note: hyphens are removed, so "High-Quality" becomes "highquality"
            expected = "highquality laptop model gb ram"
            assert cleaned_text == expected
    
    def test_text_cleaning_configuration_validation(self):
        """Test that invalid text cleaning options are rejected."""
        with pytest.raises(ValueError, match="Invalid text cleaning option"):
            ProcessingConfig(
                deduplicate_text=True,
                preserve_original_indices=True,
                text_cleaning=True,
                text_cleaning_options={
                    "invalid_option": True,  # This should cause an error
                    "lowercase": True
                }
            )
    
    def test_text_processor_methods(self):
        """Test individual TextProcessor methods."""
        processor = TextProcessor()
        
        # Test individual cleaning methods
        assert processor._to_lowercase("Hello World") == "hello world"
        assert processor._remove_punctuation("Hello, World!") == "Hello World"
        assert processor._remove_numbers("Product 123") == "Product "
        assert processor._normalize_whitespace("  multiple    spaces  ") == "multiple spaces"
        
        # Test the main clean_text method
        assert processor.clean_text(
            "  Hello, World! 123  ",
            remove_punctuation=True,
            lowercase=True,
            remove_numbers=True,
            remove_extra_whitespace=True
        ) == "hello world"
    
    def test_configuration_statistics_inclusion(self, basic_config):
        """Test that text cleaning options are included in statistics."""
        with patch('cocoon.core.vectorization.base.LocalFileReader'), \
             patch('cocoon.core.vectorization.base.LocalVectorStorage'), \
             patch.object(DatabaseVectorizer, '_create_embeddings', return_value=Mock()):
            
            vectorizer = DatabaseVectorizer(basic_config)
            stats = vectorizer.get_statistics()
            
            assert 'text_cleaning_options' in stats
            assert stats['text_cleaning_options'] == basic_config.processing.text_cleaning_options
            assert stats['text_cleaning_enabled'] == True
