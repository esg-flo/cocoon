"""Service for interpreting product information using LLM analysis."""

import json
import re
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from .core.llm.base import LLM
from .core.messages import SystemMessage, HumanMessage
from .utils.logging import logger


class ProductInterpretationService:
    """Service for interpreting product information using LLM analysis."""
    
    def __init__(self, prompt_file_path: Optional[str] = None):
        """Initialize the Product Interpretation Service.
        
        Args:
            prompt_file_path: Path to the prompt file (defaults to core/prompts/product_understanding_prompt.txt)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load prompt template
        self.prompt_file_path = prompt_file_path or Path(__file__).parent / "core" / "prompts" / "product_understanding_prompt.txt"
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        try:
            with open(self.prompt_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Prompt file not found: {self.prompt_file_path}")
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file_path}")
        except Exception as e:
            self.logger.error(f"Error loading prompt file: {e}")
            raise
    
    def _format_prompt(self, product_data: Dict[str, str]) -> str:
        """Format the prompt with product data."""
        return self.prompt_template.format(
            productGrouping=product_data.get("productGrouping", "-"),
            description1=product_data.get("description1", "-"),
            description2=product_data.get("description2", "-"),
            description3=product_data.get("description3", "-")
        )
    
    def _extract_summary_display_regex(self, response_text: str) -> str:
        """Extract summary_display using regex fallback when JSON parsing fails."""
        try:
            # Try to find summary_display field in the text
            # Look for patterns like "summary_display": "..." or summary_display: ... or similar
            patterns = [
                r'"summary_display["\s]*:["\s]*["\']([^"\']+)["\']',
                r'"summary_display["\s]*:["\s]*([^,\n\r}]+)["]',
                r'"summary_display["\s]*["\']([^"\']+)["\']',
                r'"summary_display["\s]*([^,\n\r}]+)["]'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    summary = match.group(1).strip()
                    if summary and len(summary) > 10:  # Basic validation that we got meaningful content
                        return summary
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Regex fallback failed: {e}")
            return ""
    
    def _parse_llm_response(self, response_text: str) -> str:
        """Parse the LLM response and extract summary_display using three-tier fallback."""
        try:
            # Tier 1: Try ast.literal_eval first (handles Python dict literals)
            try:
                parsed_response = ast.literal_eval(response_text.strip())
                if isinstance(parsed_response, dict) and "summary_display" in parsed_response:
                    return parsed_response["summary_display"]
                elif isinstance(parsed_response, dict):
                    self.logger.warning("summary_display field not found in parsed dict, trying JSON fallback")
                else:
                    self.logger.warning("Response is not a dict, trying JSON fallback")
            except (ValueError, SyntaxError) as e:
                self.logger.debug(f"ast.literal_eval failed: {e}, trying JSON fallback")
            
            # Tier 2: Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                try:
                    parsed_response = json.loads(json_str)
                    if "summary_display" in parsed_response:
                        return parsed_response["summary_display"]
                    else:
                        self.logger.warning("summary_display field not found in JSON, using regex fallback")
                except json.JSONDecodeError as e:
                    self.logger.debug(f"JSON parsing failed: {e}, using regex fallback")
            else:
                self.logger.debug("No JSON brackets found, using regex fallback")
            
            # Tier 3: Regex fallback
            return self._extract_summary_display_regex(response_text)
            
        except Exception as e:
            self.logger.warning(f"Unexpected error in _parse_llm_response: {e}, using regex fallback")
            return self._extract_summary_display_regex(response_text)
    
    def interpret_single_product(self, product_data: Dict[str, str], llm: LLM) -> str:
        """Interpret a single product using the LLM.
        
        Args:
            product_data: Dictionary containing product information with keys:
                - productGrouping: Product grouping/category
                - description1: Primary description
                - description2: Secondary description (optional)
                - description3: Tertiary description (optional)
            llm: LLM instance to use for interpretation
        
        Returns:
            String containing the interpretation result
        """
        try:
            # Format prompt with product data
            formatted_prompt = self._format_prompt(product_data)
            
            # Prepare messages for LLM
            messages = [
                ("system", "You are an industrial taxonomy assistant. Follow the instructions exactly and return only valid JSON."),
                ("human", formatted_prompt)
            ]
            
            # Call LLM
            self.logger.debug(f"Calling LLM for product: {product_data.get('productGrouping', 'Unknown')}")
            response_text, response_body, llm_output = llm.invoke(messages)
            
            # Parse and extract summary_display
            interpretation_result = self._parse_llm_response(response_text)
            
            self.logger.debug(f"Successfully interpreted product")
            return interpretation_result
            
        except Exception as e:
            self.logger.error(f"Failed to interpret product {product_data.get('productGrouping', 'Unknown')}: {e}")
            return ""
    
    def interpret_dataframe(
        self, 
        df: pd.DataFrame, 
        llm: LLM,
        product_grouping_col: str = "productGrouping",
        description_cols: Optional[List[str]] = None,
        output_col: str = "ai_interpretation_of_product",
        batch_size: int = 10
    ) -> pd.DataFrame:
        """Interpret all products in a dataframe.
        
        Args:
            df: Input dataframe containing product information
            llm: LLM instance to use for interpretation
            product_grouping_col: Name of the column containing product grouping
            description_cols: List of column names for descriptions (defaults to ["description1", "description2", "description3"])
            output_col: Name of the column to store the interpretation results (defaults to "ai_interpretation_of_product")
            batch_size: Number of products to process in each batch
        
        Returns:
            DataFrame with original columns plus the specified output column
        """
        if description_cols is None:
            description_cols = ["description1", "description2", "description3"]
        
        # Validate required columns exist
        required_cols = [product_grouping_col] + description_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create copy of dataframe to avoid modifying original
        result_df = df.copy()
        
        # Add output column if it doesn't exist
        if output_col not in result_df.columns:
            result_df[output_col] = ""
        
        # Identify rows that need processing (empty or missing values in output column)
        pending_mask = (
            result_df[output_col].isna() | 
            (result_df[output_col] == "") |
            (result_df[output_col].astype(str).str.strip() == "")
        )
        pending_indices = result_df[pending_mask].index.tolist()
        
        total_products = len(df)
        pending_count = len(pending_indices)
        
        if pending_count == 0:
            self.logger.info(f"All products already have {output_col} values. No processing needed.")
            return result_df
        
        self.logger.info(f"Starting interpretation of {pending_count} pending products out of {total_products} total products")
        self.logger.info(f"Processing in batches of {batch_size}")
        
        # Process pending rows in batches
        processed_count = 0
        for i in range(0, len(pending_indices), batch_size):
            batch_end = min(i + batch_size, len(pending_indices))
            batch_indices = pending_indices[i:batch_end]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}: rows {i+1}-{batch_end} of pending items")
            
            for idx in batch_indices:
                try:
                    row = result_df.loc[idx]
                    # Prepare product data
                    product_data = {
                        "productGrouping": str(row[product_grouping_col]) if pd.notna(row[product_grouping_col]) else "-",
                        "description1": str(row[description_cols[0]]) if len(description_cols) > 0 and pd.notna(row[description_cols[0]]) else "-",
                        "description2": str(row[description_cols[1]]) if len(description_cols) > 1 and pd.notna(row[description_cols[1]]) else "-",
                        "description3": str(row[description_cols[2]]) if len(description_cols) > 2 and pd.notna(row[description_cols[2]]) else "-"
                    }
                    
                    # Interpret product
                    summary_display = self.interpret_single_product(product_data, llm)
                    
                    # Update result dataframe
                    result_df.at[idx, output_col] = summary_display
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process row {idx}: {e}")
                    # Set empty value for failed rows
                    result_df.at[idx, output_col] = ""
                    processed_count += 1
            
            # Log progress
            remaining = len(pending_indices) - processed_count
            self.logger.info(f"Processed {processed_count}/{len(pending_indices)} pending products. {remaining} remaining...")
        
        self.logger.info(f"Completed interpretation of {processed_count} pending products")
        return result_df
