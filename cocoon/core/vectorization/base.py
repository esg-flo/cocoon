"""Core vectorization service for the Database Vectorizer."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from ..config.models import VectorizationPipelineConfig
from ..storage.base import FileReader, VectorStorage
from ..storage.local import LocalFileReader, LocalVectorStorage
from ..storage.s3 import S3FileReader, S3VectorStorage
from ..embeddings.base import Embeddings
from .processors import TextProcessor


class DatabaseVectorizer:
    """Core vectorization service for processing database files."""
    
    def __init__(self, config: VectorizationPipelineConfig):
        """Initialize the Database Vectorizer.
        
        Args:
            config: Complete vectorization pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage components
        self.file_reader = self._create_file_reader()
        self.vector_storage = self._create_vector_storage()
        
        # Initialize embeddings
        self.embeddings = self._create_embeddings()
        
        # Initialize text processor
        self.text_processor = TextProcessor()
        
        # Validate configuration
        self.config.validate_pipeline()
    
    def _create_file_reader(self) -> FileReader:
        """Create appropriate file reader based on configuration."""
        if self.config.input.is_s3_path:
            return S3FileReader()
        else:
            return LocalFileReader()
    
    def _create_vector_storage(self) -> VectorStorage:
        """Create appropriate vector storage based on configuration."""
        if self.config.output.is_s3_output:
            return S3VectorStorage()
        else:
            return LocalVectorStorage()
    
    def _create_embeddings(self) -> Embeddings:
        """Create embeddings instance based on configuration."""
        # Import here to avoid circular imports
        from ..embeddings.bedrock import BedrockEmbeddings
        
        embedding_config = self.config.vectorization.embedding_model_config
        
        if 'model_id' in embedding_config:
            return BedrockEmbeddings(**embedding_config)
        else:
            # Use default configuration
            return BedrockEmbeddings(
                model_id=os.getenv("EMBEDDING_MODEL_ID"),
                aws_region_name=os.getenv("AWS_REGION_NAME")
            )
    
    def process(self) -> pd.DataFrame:
        """Process the input file and return vectorized data.
        
        Returns:
            DataFrame with columns: label, embedding, index_ids, + metadata columns
        """
        self.logger.info(f"Starting vectorization of {self.config.input.file_path}")
        
        # Step 1: Read input file
        df = self._read_input_file()
        self.logger.info(f"Loaded {len(df)} rows from input file")
        
        # Step 2: Process text data
        processed_df = self._process_text_data(df)
        self.logger.info(f"Processed {len(processed_df)} unique text entries")
        
        # Step 3: Generate embeddings
        vectorized_df = self._generate_embeddings(processed_df)
        self.logger.info(f"Generated embeddings for {len(vectorized_df)} entries")
        
        # Step 4: Save output
        self._save_output(vectorized_df)
        self.logger.info(f"Saved output to {self.config.output.output_path}")
        
        return vectorized_df
    
    def _read_input_file(self) -> pd.DataFrame:
        """Read the input file based on its type."""
        file_path = self.config.input.file_path
        file_type = self.config.input.detected_file_type
        
        if file_type == 'csv':
            return self.file_reader.read_csv(
                file_path,
                delimiter=self.config.input.csv_delimiter,
                encoding=self.config.input.csv_encoding
            )
        elif file_type == 'excel':
            return self.file_reader.read_excel(
                file_path,
                sheet_name=self.config.input.excel_sheet_name,
                engine=self.config.input.excel_engine
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process text data including deduplication and cleaning."""
        processing_config = self.config.processing
        vectorization_config = self.config.vectorization
        target_column = vectorization_config.target_column
        metadata_columns = vectorization_config.metadata_columns
        
        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in input file")
        
        # Validate metadata columns exist
        missing_metadata = [col for col in metadata_columns if col not in df.columns]
        if missing_metadata:
            raise ValueError(f"Metadata columns not found in input file: {missing_metadata}")

        # Create a copy for processing with only specified columns
        columns_to_keep = [target_column] + metadata_columns
        processed_df = df[columns_to_keep].copy()

        # Filter out empty rows in the target column if enabled
        if processing_config.filter_empty_rows:
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna(subset=[target_column])
            processed_df = processed_df[processed_df[target_column].astype(str).str.strip() != '']
            filtered_rows = len(processed_df)
            
            if initial_rows != filtered_rows:
                self.logger.info(f"Filtered out {initial_rows - filtered_rows} empty rows from target column '{target_column}'")
        
        # Store original text for the label column
        processed_df['label'] = processed_df[target_column].copy()
        
        # Apply text cleaning if enabled - create a separate column for embedding
        if processing_config.text_cleaning:
            cleaning_options = processing_config.text_cleaning_options
            processed_df['_cleaned_text'] = processed_df[target_column].apply(
                lambda text: self.text_processor.clean_text_with_config(text, cleaning_options)
            )
        else:
            # If no cleaning, use original text for embedding
            processed_df['_cleaned_text'] = processed_df[target_column].copy()
        
        # Deduplicate text if enabled
        if processing_config.deduplicate_text:
            processed_df = self._deduplicate_text(processed_df, '_cleaned_text', processing_config)
        else:
            # If no deduplication, just rename the cleaned text column for consistency
            processed_df = processed_df.rename(columns={'_cleaned_text': '_text_for_embedding'})
        
        return processed_df

    
    def _deduplicate_text(self, df: pd.DataFrame, text_column: str, processing_config) -> pd.DataFrame:
        """Deduplicate text while preserving row indices."""
        # Create a copy and add the index as a column for aggregation
        df_copy = df.copy()
        df_copy['_temp_index'] = df_copy.index
        
        # Group by cleaned text content and collect row indices
        grouped = df_copy.groupby(text_column).agg({
            **{col: 'first' for col in df.columns if col not in [text_column, '_temp_index']},
            '_temp_index': lambda x: list(x)
        }).reset_index()
        
        # Rename the cleaned text column to indicate it's for embedding
        grouped = grouped.rename(columns={text_column: '_text_for_embedding'})
        
        # Create index_ids column
        if processing_config.preserve_original_indices:
            grouped['index_ids'] = grouped['_temp_index']
        else:
            grouped['index_ids'] = grouped['_temp_index'].apply(lambda x: [i for i in range(len(x))])
        
        # Drop the temporary index column
        grouped = grouped.drop('_temp_index', axis=1)
        
        return grouped
    
    def _generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for the text data."""
        vectorization_config = self.config.vectorization
        batch_size = vectorization_config.batch_size
        
        # Use the cleaned text for embedding generation
        text_column = '_text_for_embedding' if '_text_for_embedding' in df.columns else 'label'
        texts = df[text_column].tolist()
        
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            
            self.logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
        
        # Add embeddings column
        df['embedding'] = embeddings
        
        # Clean up temporary columns and reorder
        if '_text_for_embedding' in df.columns:
            df = df.drop('_text_for_embedding', axis=1)
        
        # Reorder columns: label, embedding, index_ids, then metadata
        column_order = ['label', 'embedding', 'index_ids']
        metadata_columns = [col for col in df.columns if col not in column_order]
        df = df[column_order + metadata_columns]
        
        return df
    
    def _save_output(self, df: pd.DataFrame) -> None:
        """Save the vectorized data to the specified output format."""
        output_config = self.config.output
        output_path = output_config.output_path
        output_format = output_config.output_format
        
        # Prepare save kwargs
        save_kwargs = {}
        
        if output_format == 'parquet':
            if output_config.compression:
                save_kwargs['compression'] = output_config.compression
            self.vector_storage.save_parquet(df, output_path, **save_kwargs)
            
        elif output_format == 'csv':
            self.vector_storage.save_csv(df, output_path, **save_kwargs)
            
        elif output_format == 'jsonl':
            self.vector_storage.save_jsonl(df, output_path, **save_kwargs)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'input_file': self.config.input.file_path,
            'output_file': self.config.output.output_path,
            'input_format': self.config.input.detected_file_type,
            'output_format': self.config.output.output_format,
            'target_column': self.config.vectorization.target_column,
            'metadata_columns': self.config.vectorization.metadata_columns,
            'batch_size': self.config.vectorization.batch_size,
            'deduplication_enabled': self.config.processing.deduplicate_text,
            'text_cleaning_enabled': self.config.processing.text_cleaning,
            'text_cleaning_options': self.config.processing.text_cleaning_options
        }
