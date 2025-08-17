"""Unit tests for configuration models."""

import pytest
from pathlib import Path
from cocoon.core.config.models import (
    FileInputConfig,
    ProcessingConfig,
    VectorizationConfig,
    OutputConfig,
    VectorizationPipelineConfig,
)


class TestFileInputConfig:
    """Test FileInputConfig class."""
    
    def test_csv_file_auto_detection(self):
        """Test CSV file type auto-detection."""
        config = FileInputConfig(file_path="data/products.csv")
        assert config.detected_file_type == "csv"
        assert not config.is_s3_path
        assert config.local_path == Path("data/products.csv")
    
    def test_excel_file_auto_detection(self):
        """Test Excel file type auto-detection."""
        config = FileInputConfig(file_path="data/products.xlsx")
        assert config.detected_file_type == "excel"
        assert not config.is_s3_path
    
    def test_excel_file_auto_detection_xls(self):
        """Test Excel .xls file type auto-detection."""
        config = FileInputConfig(file_path="data/products.xls")
        assert config.detected_file_type == "excel"
    
    def test_s3_path_detection(self):
        """Test S3 path detection."""
        config = FileInputConfig(file_path="s3://bucket/data.csv")
        assert config.is_s3_path
        assert config.local_path is None
    
    def test_explicit_file_type(self):
        """Test explicit file type override."""
        config = FileInputConfig(file_path="data/products.csv", file_type="excel")
        assert config.file_type == "excel"
    
    def test_invalid_file_extension(self):
        """Test invalid file extension raises error."""
        config = FileInputConfig(file_path="data/products.txt")
        with pytest.raises(ValueError, match="Cannot auto-detect file type"):
            _ = config.detected_file_type
    
    def test_empty_file_path(self):
        """Test empty file path raises error."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            FileInputConfig(file_path="")
    
    def test_csv_configuration(self):
        """Test CSV-specific configuration."""
        config = FileInputConfig(
            file_path="data/products.csv",
            csv_delimiter=";",
            csv_encoding="latin-1"
        )
        assert config.csv_delimiter == ";"
        assert config.csv_encoding == "latin-1"
    
    def test_excel_configuration(self):
        """Test Excel-specific configuration."""
        config = FileInputConfig(
            file_path="data/products.xlsx",
            excel_sheet_name="Products",
            excel_engine="xlrd"
        )
        assert config.excel_sheet_name == "Products"
        assert config.excel_engine == "xlrd"


class TestProcessingConfig:
    """Test ProcessingConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        assert config.deduplicate_text is True
        assert config.preserve_original_indices is True
        assert config.text_cleaning is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            deduplicate_text=False,
            preserve_original_indices=False,
            text_cleaning=False
        )
        assert config.deduplicate_text is False
        assert config.preserve_original_indices is False
        assert config.text_cleaning is False
    



class TestVectorizationConfig:
    """Test VectorizationConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VectorizationConfig(target_column="description")
        assert config.target_column == "description"
        assert config.metadata_columns == []
        assert config.batch_size == 100
        assert config.embedding_model_config == {}
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = VectorizationConfig(
            target_column="product_description",
            metadata_columns=["category", "price"],
            batch_size=50,
            embedding_model_config={"model_id": "test-model"}
        )
        assert config.target_column == "product_description"
        assert config.metadata_columns == ["category", "price"]
        assert config.batch_size == 50
        assert config.embedding_model_config == {"model_id": "test-model"}
    
    def test_target_column_validation(self):
        """Test target column validation."""
        with pytest.raises(ValueError, match="Target column cannot be empty"):
            VectorizationConfig(target_column="")
        
        with pytest.raises(ValueError, match="Target column cannot be empty"):
            VectorizationConfig(target_column="   ")
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="Batch size must be at least 1"):
            VectorizationConfig(target_column="desc", batch_size=0)
        
        with pytest.raises(ValueError, match="Batch size cannot exceed 1000"):
            VectorizationConfig(target_column="desc", batch_size=1001)
    
    def test_target_column_stripping(self):
        """Test target column whitespace stripping."""
        config = VectorizationConfig(target_column="  description  ")
        assert config.target_column == "description"


class TestOutputConfig:
    """Test OutputConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OutputConfig(output_path="output/data.parquet")
        assert config.output_path == "output/data.parquet"
        assert config.output_format == "parquet"
        assert config.compression == "snappy"
        assert config.include_metadata is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = OutputConfig(
            output_path="output/data.csv",
            output_format="csv",
            compression=None,
            include_metadata=False
        )
        assert config.output_path == "output/data.csv"
        assert config.output_format == "csv"
        assert config.compression is None
        assert config.include_metadata is False
    
    def test_output_format_validation(self):
        """Test output format validation."""
        with pytest.raises(ValueError, match="Output format must be one of"):
            OutputConfig(output_path="output/data.txt", output_format="txt")
    
    def test_compression_validation(self):
        """Test compression validation for parquet."""
        with pytest.raises(ValueError, match="Compression must be one of"):
            OutputConfig(
                output_path="output/data.parquet",
                output_format="parquet",
                compression="invalid"
            )
    
    def test_compression_ignored_for_non_parquet(self):
        """Test compression is ignored for non-parquet formats."""
        config = OutputConfig(
            output_path="output/data.csv",
            output_format="csv",
            compression="gzip"
        )
        assert config.compression == "gzip"  # Should not raise error
    
    def test_s3_output_detection(self):
        """Test S3 output path detection."""
        config = OutputConfig(output_path="s3://bucket/output/data.parquet")
        assert config.is_s3_output is True
        
        config = OutputConfig(output_path="output/data.parquet")
        assert config.is_s3_output is False


class TestVectorizationPipelineConfig:
    """Test VectorizationPipelineConfig class."""
    
    def test_valid_pipeline_configuration(self):
        """Test valid pipeline configuration."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(file_path="data/products.csv"),
            processing=ProcessingConfig(),
            vectorization=VectorizationConfig(
                target_column="description",
                metadata_columns=["category", "price"]
            ),
            output=OutputConfig(output_path="output/data.parquet")
        )
        assert config.validate_pipeline() is True
    
    def test_target_column_in_metadata_validation(self):
        """Test validation that target column is not in metadata."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(file_path="data/products.csv"),
            processing=ProcessingConfig(),
            vectorization=VectorizationConfig(
                target_column="description",
                metadata_columns=["description", "category"]  # Target column in metadata
            ),
            output=OutputConfig(output_path="output/data.parquet")
        )
        
        with pytest.raises(ValueError, match="Target column cannot be in metadata columns"):
            config.validate_pipeline()
    
    def test_csv_to_excel_conversion_validation(self):
        """Test validation that CSV to Excel conversion is supported."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(file_path="data/products.csv"),
            processing=ProcessingConfig(),
            vectorization=VectorizationConfig(target_column="description"),
            output=OutputConfig(
                output_path="output/data.xlsx",
                output_format="excel"
            )
        )
        
        # Should not raise an error
        assert config.validate_pipeline() is True
    
    def test_excel_to_csv_conversion_allowed(self):
        """Test that Excel to CSV conversion is allowed."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(file_path="data/products.xlsx"),
            processing=ProcessingConfig(),
            vectorization=VectorizationConfig(target_column="description"),
            output=OutputConfig(
                output_path="output/data.csv",
                output_format="csv"
            )
        )
        
        # Should not raise an error
        assert config.validate_pipeline() is True


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""
    
    def test_csv_processing_pipeline(self):
        """Test complete CSV processing pipeline configuration."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(
                file_path="data/products.csv",
                csv_delimiter=",",
                csv_encoding="utf-8"
            ),
            processing=ProcessingConfig(
                deduplicate_text=True,
                preserve_original_indices=True,
                text_cleaning=True
            ),
            vectorization=VectorizationConfig(
                target_column="product_description",
                metadata_columns=["category", "price", "brand"],
                batch_size=100,
                embedding_model_config={
                    "model_id": "amazon.titan-embed-text-v1",
                    "aws_region_name": "us-east-1"
                }
            ),
            output=OutputConfig(
                output_path="output/vectorized_products.parquet",
                output_format="parquet",
                compression="snappy"
            )
        )
        
        assert config.validate_pipeline() is True
        assert config.input.detected_file_type == "csv"
        assert config.output.output_format == "parquet"
        assert config.vectorization.target_column == "product_description"
    
    def test_s3_excel_processing_pipeline(self):
        """Test complete S3 Excel processing pipeline configuration."""
        config = VectorizationPipelineConfig(
            input=FileInputConfig(
                file_path="s3://my-bucket/data/products.xlsx",
                excel_sheet_name="Products",
                excel_engine="openpyxl"
            ),
            processing=ProcessingConfig(
                deduplicate_text=True,
                preserve_original_indices=True
            ),
            vectorization=VectorizationConfig(
                target_column="description",
                metadata_columns=["category", "price"],
                batch_size=50
            ),
            output=OutputConfig(
                output_path="s3://my-bucket/output/vectorized_products.jsonl",
                output_format="jsonl"
            )
        )
        
        assert config.validate_pipeline() is True
        assert config.input.is_s3_path is True
        assert config.output.is_s3_output is True
        assert config.input.detected_file_type == "excel"
        assert config.output.output_format == "jsonl"
