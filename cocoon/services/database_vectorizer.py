"""High-level service for database vectorization operations."""

import logging
import os
from typing import Dict, Any, Optional

from ..core.config.models import VectorizationPipelineConfig
from ..core.vectorization.base import DatabaseVectorizer
from ..core.embeddings.bedrock import BedrockEmbeddings


class DatabaseVectorizerService:
    """High-level service for orchestrating database vectorization operations."""
    
    def __init__(self, config: Optional[VectorizationPipelineConfig] = None):
        """Initialize the Database Vectorizer Service.
        
        Args:
            config: Optional default configuration template. If not provided, 
                          will use environment-based defaults.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._create_config()
    
    def _create_config(self) -> VectorizationPipelineConfig:
        """Create default configuration from environment variables."""
        from ..core.config.models import (
            FileInputConfig, ProcessingConfig, VectorizationConfig, OutputConfig
        )
        
        # Get default embedding model from environment
        default_model = os.getenv(
            'EMBEDDING_MODEL_ID', 
            'amazon.titan-embed-text-v1'
        )
        default_region = os.getenv('AWS_DEFAULT_REGION', 'eu-central-1')
        default_batch_size = int(os.getenv('DB_VECTORIZER_DEFAULT_BATCH_SIZE', '100'))
        
        return VectorizationPipelineConfig(
            input=FileInputConfig(
                file_path="placeholder.csv",  # Placeholder that will be replaced
                file_type="csv",  # Default to CSV
                csv_delimiter=",",
                csv_encoding="utf-8",
                excel_sheet_name=None,
                excel_engine="openpyxl"
            ),
            processing=ProcessingConfig(
                deduplicate_text=True,
                preserve_original_indices=True,
                filter_empty_rows=True,
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
                embedding_model_config={
                    "model_id": default_model,
                    "aws_region_name": default_region
                },
                target_column="text",  # Default column name
                metadata_columns=[],
                batch_size=default_batch_size
            ),
            output=OutputConfig(
                output_path="output.parquet",  # Placeholder that will be replaced
                output_format="parquet",
                compression="snappy",
                include_metadata=True
            )
        )
    
    def process_file(
        self,
        input_file_path: str,
        target_column: str,
        output_path: str,
        metadata_columns: Optional[list] = None,
        output_format: str = "parquet",
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single file for vectorization.
        
        Args:
            input_file_path: Path to input file (local or S3)
            target_column: Name of the column to vectorize
            output_path: Path for output file
            metadata_columns: Optional list of metadata columns to preserve
            output_format: Output format (parquet, csv, jsonl, excel)
            **kwargs: Additional configuration overrides for processing and vectorization:
                - text_cleaning: bool - Enable/disable text cleaning
                - text_cleaning_options: dict - Detailed text cleaning options:
                    - lowercase: bool - Convert to lowercase
                    - remove_punctuation: bool - Remove punctuation marks
                    - remove_numbers: bool - Remove numeric characters
                    - remove_special_chars: bool - Remove special characters
                    - normalize_whitespace: bool - Normalize whitespace
                - deduplicate_text: bool - Enable/disable text deduplication
                - preserve_original_indices: bool - Preserve original row indices
                - batch_size: int - Batch size for embedding generation
            
        Returns:
            Dictionary containing processing results and statistics
        """
        try:
            self.logger.info(f"Starting vectorization of {input_file_path}")
            
            # Create a new configuration for this specific file
            config = self._create_file_config(
                input_file_path, target_column, output_path, 
                metadata_columns, output_format, **kwargs
            )
            
            # Validate the configuration
            config.validate_pipeline()
            
            # Create a new vectorizer instance with this config
            vectorizer = DatabaseVectorizer(config)
            
            # Process the file
            result_df = vectorizer.process()
            
            # Collect and return statistics
            stats = vectorizer.get_statistics()
            stats.update({
                'rows_processed': len(result_df),
                'unique_texts': len(result_df),
                'status': 'success'
            })
            
            self.logger.info(f"Successfully processed {input_file_path}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error processing {input_file_path}: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__,
                'input_file': input_file_path,
                'target_column': target_column,
                'output_path': output_path
            }
    
    def _create_file_config(
        self,
        input_file_path: str,
        target_column: str,
        output_path: str,
        metadata_columns: Optional[list] = None,
        output_format: str = "parquet",
        **kwargs
    ) -> VectorizationPipelineConfig:
        """Create a new configuration instance for file processing.
        
        This method creates a fresh configuration for each file to ensure
        no shared state between different processing operations.
        """
        # Create a deep copy of the default configuration
        config = self.config.model_copy(deep=True)
        
        # Update input configuration
        config.input.file_path = input_file_path
        
        # Update vectorization configuration
        config.vectorization.target_column = target_column
        if metadata_columns is not None:
            config.vectorization.metadata_columns = metadata_columns
        
        # Update output configuration
        config.output.output_path = output_path
        config.output.output_format = output_format
        
        # Apply any additional overrides
        for key, value in kwargs.items():
            if hasattr(config.processing, key):
                if key == 'text_cleaning_options' and isinstance(value, dict):
                    # Merge text cleaning options instead of replacing
                    current_options = config.processing.text_cleaning_options.copy()
                    current_options.update(value)
                    setattr(config.processing, key, current_options)
                else:
                    setattr(config.processing, key, value)
            elif hasattr(config.vectorization, key):
                setattr(config.vectorization, key, value)
            elif hasattr(config.output, key):
                setattr(config.output, key, value)
        
        return config
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the service.
        
        Returns:
            Dictionary containing health status and configuration validation
        """
        try:
            # Check if embeddings can be created
            test_embeddings = BedrockEmbeddings(
                model_id=os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1"),
                aws_region_name=os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
            )
            
            # Test with a simple text
            test_embedding = test_embeddings.embed_query("test")
            
            return {
                'status': 'healthy',
                'embeddings_available': True,
                'test_embedding_dimensions': len(test_embedding),
                'configuration_valid': True,
                'config_loaded': self.config is not None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'embeddings_available': False,
                'configuration_valid': False,
                'config_loaded': self.config is not None
            }
