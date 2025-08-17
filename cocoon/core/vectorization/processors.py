"""Text processing utilities for the Database Vectorizer."""

import re
import string
import pandas as pd


class TextProcessor:
    """Text processing utilities for cleaning and normalizing text."""
    
    def _validate_text(self, text) -> str:
        """Validate and convert text input."""
        if pd.isna(text) or text is None:
            return ""
        return str(text).strip()
    
    def _to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters from text, keeping only alphanumeric characters and spaces."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return ' '.join(text.split())
    
    def clean_text(self, text: str, 
                   remove_punctuation: bool = False,
                   lowercase: bool = True,
                   remove_numbers: bool = False,
                   remove_extra_whitespace: bool = True) -> str:
        """Clean and normalize text.
        
        Args:
            text: Input text to clean
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
            remove_numbers: Whether to remove numbers
            remove_extra_whitespace: Whether to normalize whitespace
            
        Returns:
            Cleaned text
        """
        text = self._validate_text(text)
        if not text:
            return text
        
        if lowercase:
            text = self._to_lowercase(text)
        
        if remove_punctuation:
            text = self._remove_punctuation(text)
        
        if remove_numbers:
            text = self._remove_numbers(text)
        
        if remove_extra_whitespace:
            text = self._normalize_whitespace(text)
        
        return text

    def clean_text_with_config(self, text: str, config: dict) -> str:
        """Clean text using a configuration dictionary.
        
        Args:
            text: Input text to clean
            config: Dictionary with cleaning options:
                - lowercase: bool
                - remove_punctuation: bool
                - remove_numbers: bool
                - remove_special_chars: bool
                - normalize_whitespace: bool
                
        Returns:
            Cleaned text
        """
        return self.clean_text(
            text,
            remove_punctuation=config.get("remove_punctuation", False),
            lowercase=config.get("lowercase", True),
            remove_numbers=config.get("remove_numbers", False),
            remove_extra_whitespace=config.get("normalize_whitespace", True)
        )
