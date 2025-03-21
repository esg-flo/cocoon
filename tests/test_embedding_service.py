import unittest
from unittest.mock import patch

import pandas as pd
from numpy.random import random
from pandas.testing import assert_frame_equal

from cocoon.core.embeddings.bedrock import BedrockEmbeddings
from cocoon.embedding_service import create_embeddings, initialize_faiss_index_from_embeddings


class TestBedrockEmbeddings(unittest.TestCase):
    @patch("boto3.client")
    def test_create_embeddings(self, mock_client):
        model_id = "amazon.titan-v1"
        aws_region_name = "eu-central-1"
        client = mock_client

        df = pd.DataFrame(
            {
                "label": [
                    "Abrasive Product Manufacturing",
                    "Adhesive Manufacturing",
                    "Advertising Agencies",
                    "Advertising Material Distribution Services",
                    "Abrasive Product Manufacturing",
                ]
            }
        )

        expected_output = pd.DataFrame(
            {
                "label": [
                    "Abrasive Product Manufacturing",
                    "Adhesive Manufacturing",
                    "Advertising Agencies",
                    "Advertising Material Distribution Services",
                ],
                "index_ids": [[0, 4], [1], [2], [3]],
                "embedding": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            }
        )

        output_csv_filepath = "test.csv"

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, client)
        with patch(
            "cocoon.core.embeddings.bedrock.BedrockEmbeddings.embed_query"
        ) as mock_embed_query:
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                mock_embed_query.return_value = [1, 2, 3]
                output = create_embeddings(bedrock_embeddings, df, output_csv_filepath)

                assert_frame_equal(expected_output, output)
                mock_to_csv.assert_called_once_with(output_csv_filepath, index=False)

    @patch("boto3.client")
    def test_create_embeddings_partial_complete(self, mock_client):
        model_id = "amazon.titan-v1"
        aws_region_name = "eu-central-1"
        client = mock_client

        df = pd.DataFrame(
            {
                "label": [
                    "Abrasive Product Manufacturing",
                    "Adhesive Manufacturing",
                    "Advertising Agencies",
                    "Advertising Material Distribution Services",
                ],
                "index_ids": [[0, 4], [1], [2], [3]],
                "embedding": [[1, 2, 3], [1, 2, 3], None, None],
            }
        )

        expected_output = pd.DataFrame(
            {
                "label": [
                    "Abrasive Product Manufacturing",
                    "Adhesive Manufacturing",
                    "Advertising Agencies",
                    "Advertising Material Distribution Services",
                ],
                "index_ids": [[0, 4], [1], [2], [3]],
                "embedding": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            }
        )

        output_csv_filepath = "test.csv"

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, client)
        with patch(
            "cocoon.core.embeddings.bedrock.BedrockEmbeddings.embed_query"
        ) as mock_embed_query:
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("pandas.read_csv") as mock_read_csv:
                    mock_read_csv.return_value = df
                    mock_embed_query.return_value = [1, 2, 3]
                    output = create_embeddings(bedrock_embeddings, df, output_csv_filepath)

                    assert_frame_equal(expected_output, output)
                    mock_to_csv.assert_called_once_with(output_csv_filepath, index=False)

    @patch("boto3.client")
    def test_create_embeddings_full_complete(self, mock_client):
        model_id = "amazon.titan-v1"
        aws_region_name = "eu-central-1"

        df = pd.DataFrame(
            {
                "label": [
                    "Abrasive Product Manufacturing",
                    "Adhesive Manufacturing",
                    "Advertising Agencies",
                    "Advertising Material Distribution Services",
                ],
                "index_ids": [[0, 4], [1], [2], [3]],
                "embedding": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            }
        )

        expected_output = pd.DataFrame(
            {
                "label": [
                    "Abrasive Product Manufacturing",
                    "Adhesive Manufacturing",
                    "Advertising Agencies",
                    "Advertising Material Distribution Services",
                ],
                "index_ids": [[0, 4], [1], [2], [3]],
                "embedding": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            }
        )

        output_csv_filepath = "test.csv"

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client)
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = df

            output = create_embeddings(bedrock_embeddings, df, output_csv_filepath)

            assert_frame_equal(expected_output, output)

    @patch("boto3.client")
    def test_initialize_faiss_index_from_embeddings(self, mock_client):
        model_id = "amazon.titan-v1"
        aws_region_name = "eu-central-1"
        dims = 3

        df = pd.DataFrame(
            {
                "embedding": [[1.1, 2.2, 3.1], [0.1, 2.3, 0.3], [2.1, 1.2, 0.3]],
            }
        )

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client, dims=dims)

        index = initialize_faiss_index_from_embeddings(bedrock_embeddings, df)
        self.assertIsNotNone(index)

    @patch("boto3.client")
    def test_initialize_faiss_index_from_embeddings_invalid_input(self, mock_client):
        model_id = "amazon.titan-v1"
        aws_region_name = "eu-central-1"
        dims = 3

        df = pd.DataFrame({"embedding": [1, 2, 3]})

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client, dims=dims)
        with self.assertRaises(ValueError):
            initialize_faiss_index_from_embeddings(bedrock_embeddings, df)
