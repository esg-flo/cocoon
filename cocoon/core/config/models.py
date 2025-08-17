"""Configuration models for the Database Vectorizer."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class FileInputConfig(BaseModel):
    """Configuration for file input operations."""
    
    file_path: str = Field(..., description="Path to the input file (local or S3)")
    file_type: Optional[str] = Field(None, description="File type (csv, excel, auto-detected if None)")
    csv_delimiter: str = Field(",", description="CSV delimiter character")
    csv_encoding: str = Field("utf-8", description="CSV file encoding")
    excel_sheet_name: Optional[str] = Field(None, description="Excel sheet name (defaults to first sheet)")
    excel_engine: str = Field("openpyxl", description="Excel engine (openpyxl or xlrd)")
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path format."""
        if not v:
            raise ValueError("File path cannot be empty")
        return v
    
    @property
    def detected_file_type(self) -> str:
        """Get the detected file type (either explicit or auto-detected)."""
        if self.file_type is not None:
            return self.file_type.lower()
        
        path = Path(self.file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return 'csv'
        elif suffix in ['.xlsx', '.xls']:
            return 'excel'
        else:
            raise ValueError(f"Cannot auto-detect file type for extension: {suffix}")
    
    @property
    def is_s3_path(self) -> bool:
        """Check if the file path is an S3 path."""
        return self.file_path.startswith('s3://')
    
    @property
    def local_path(self) -> Optional[Path]:
        """Get local path if applicable."""
        if not self.is_s3_path:
            return Path(self.file_path)
        return None


class ProcessingConfig(BaseModel):
    """Configuration for text processing operations."""
    
    deduplicate_text: bool = Field(True, description="Whether to deduplicate text values")
    preserve_original_indices: bool = Field(True, description="Whether to preserve original row indices")
    filter_empty_rows: bool = Field(True, description="Whether to filter by text length")
    
    # Text cleaning configuration
    text_cleaning: bool = Field(True, description="Whether to apply text cleaning")
    text_cleaning_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "lowercase": True,
            "remove_punctuation": False,
            "remove_numbers": False,
            "remove_special_chars": False,
            "normalize_whitespace": True
        },
        description="Detailed text cleaning options"
    )
    
    @field_validator('text_cleaning_options')
    @classmethod
    def validate_text_cleaning_options(cls, v):
        """Validate text cleaning options."""
        valid_options = {
            "lowercase", "remove_punctuation", "remove_numbers", 
            "remove_special_chars", "normalize_whitespace"
        }
        
        for key in v:
            if key not in valid_options:
                raise ValueError(f"Invalid text cleaning option: {key}. Valid options: {valid_options}")
        
        return v


class VectorizationConfig(BaseModel):
    """Configuration for vectorization operations."""
    
    embedding_model_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the embedding model"
    )
    target_column: str = Field(..., description="Name of the column to vectorize")
    metadata_columns: List[str] = Field(
        default_factory=list,
        description="List of metadata columns to preserve"
    )
    batch_size: int = Field(100, description="Batch size for embedding generation")
    
    @field_validator('target_column')
    @classmethod
    def validate_target_column(cls, v):
        """Validate target column name."""
        if not v or not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        if v > 1000:
            raise ValueError("Batch size cannot exceed 1000")
        return v


class OutputConfig(BaseModel):
    """Configuration for output operations."""
    
    output_path: str = Field(..., description="Path for the output file")
    output_format: str = Field("parquet", description="Output format (parquet, csv, jsonl)")
    compression: Optional[str] = Field("snappy", description="Compression for parquet files")
    include_metadata: bool = Field(True, description="Whether to include metadata columns")
    
    @field_validator('output_format')
    @classmethod
    def validate_output_format(cls, v):
        """Validate output format."""
        valid_formats = ['parquet', 'csv', 'jsonl', 'excel']
        if v.lower() not in valid_formats:
            raise ValueError(f"Output format must be one of: {valid_formats}")
        return v.lower()
    
    @field_validator('compression')
    @classmethod
    def validate_compression(cls, v, info):
        """Validate compression setting."""
        if v is not None and info.data.get('output_format') == 'parquet':
            valid_compressions = ['snappy', 'gzip', 'brotli', 'lz4', 'zstd']
            if v.lower() not in valid_compressions:
                raise ValueError(f"Compression must be one of: {valid_compressions}")
            return v.lower()
        return v
    
    @property
    def is_s3_output(self) -> bool:
        """Check if the output path is an S3 path."""
        return self.output_path.startswith('s3://')


class VectorizationPipelineConfig(BaseModel):
    """Complete configuration for the vectorization pipeline."""
    
    input: FileInputConfig
    processing: ProcessingConfig
    vectorization: VectorizationConfig
    output: OutputConfig
    
    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }
    
    def validate_pipeline(self) -> bool:
        """Validate the complete pipeline configuration."""
        # Check if target column exists in metadata columns
        if self.vectorization.target_column in self.vectorization.metadata_columns:
            raise ValueError("Target column cannot be in metadata columns")
        
        # Check if output format is compatible with input format
        input_file_type = self.input.detected_file_type
        if input_file_type == 'excel' and self.output.output_format == 'csv':
            # Excel to CSV conversion is supported
            pass
        elif input_file_type == 'csv' and self.output.output_format == 'excel':
            # CSV to Excel conversion is supported
            pass
        
        return True
