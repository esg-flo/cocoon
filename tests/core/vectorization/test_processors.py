"""Unit tests for text processing utilities."""

import pytest
import pandas as pd
from cocoon.core.vectorization.processors import TextProcessor


class TestTextProcessor:
    """Test TextProcessor class."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        processor = TextProcessor()
        text = "  Hello   World!  "
        result = processor.clean_text(text)
        assert result == "hello world!"
    
    def test_clean_text_remove_punctuation(self):
        """Test text cleaning with punctuation removal."""
        processor = TextProcessor()
        text = "Hello, World! How are you?"
        result = processor.clean_text(text, remove_punctuation=True)
        assert result == "hello world how are you"
    
    def test_clean_text_remove_numbers(self):
        """Test text cleaning with number removal."""
        processor = TextProcessor()
        text = "Hello 123 World 456"
        result = processor.clean_text(text, remove_numbers=True)
        assert result == "hello world"
    
    def test_clean_text_keep_original_case(self):
        """Test text cleaning without lowercase conversion."""
        processor = TextProcessor()
        text = "Hello WORLD"
        result = processor.clean_text(text, lowercase=False)
        assert result == "Hello WORLD"
    
    def test_clean_text_none_input(self):
        """Test text cleaning with None input."""
        processor = TextProcessor()
        result = processor.clean_text(None)
        assert result == ""
    
    def test_clean_text_pandas_na(self):
        """Test text cleaning with pandas NA."""
        processor = TextProcessor()
        result = processor.clean_text(pd.NA)
        assert result == ""
    
    def test_clean_text_with_config(self):
        """Test text cleaning with configuration dictionary."""
        processor = TextProcessor()
        text = "  Hello, World! 123  "
        config = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": True,
            "remove_special_chars": False,
            "normalize_whitespace": True
        }
        result = processor.clean_text_with_config(text, config)
        assert result == "hello world"
    
    def test_clean_text_with_config_partial(self):
        """Test text cleaning with partial configuration."""
        processor = TextProcessor()
        text = "  Hello, World!  "
        config = {
            "lowercase": False,
            "normalize_whitespace": True
        }
        result = processor.clean_text_with_config(text, config)
        assert result == "Hello, World!"
    
    def test_clean_text_empty_string(self):
        """Test text cleaning with empty string."""
        processor = TextProcessor()
        text = ""
        result = processor.clean_text(text)
        assert result == ""
    
    def test_clean_text_whitespace_only(self):
        """Test text cleaning with whitespace-only string."""
        processor = TextProcessor()
        text = "   \n\t   "
        result = processor.clean_text(text)
        assert result == ""
