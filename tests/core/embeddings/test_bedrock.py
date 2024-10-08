import unittest
from unittest.mock import patch

from numpy.random import random

from src.core.embeddings.bedrock import BedrockEmbeddings


class TestBedrockEmbeddings(unittest.TestCase):
    @patch("boto3.client")
    def test_init(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "eu-central-1"
        client = mock_client
        normalize = False
        dims = 256

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, client, normalize, dims)

        self.assertEqual(bedrock_embeddings.client, mock_client)
        self.assertEqual(bedrock_embeddings.model_id, model_id)
        self.assertEqual(bedrock_embeddings.normalize, normalize)
        self.assertEqual(bedrock_embeddings.dims, dims)

    @patch("boto3.client")
    def test_dimensions(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "eu-central-1"
        dims = 256

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client, dims=dims)

        self.assertEqual(bedrock_embeddings.dimensions, dims)

    @patch("boto3.client")
    def test_model_id(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "eu-central-1"
        dims = 256

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client, dims=dims)

        self.assertEqual(bedrock_embeddings.model_id, model_id)

    @patch("boto3.client")
    def test_embedding_func(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "eu-central-1"
        dims = 256

        embeddings = random(size=dims)
        mock_client.return_value.invoke_model.return_value = {"body": b'{"embedding": {}}'}
        with patch("json.loads") as mock_json_loads:
            mock_json_loads.return_value = {"embedding": embeddings}

            bedrock_embeddings = BedrockEmbeddings(
                model_id, aws_region_name, mock_client, dims=dims
            )

            self.assertEqual(len(bedrock_embeddings._embedding_func("test-text")), len(embeddings))

    @patch("boto3.client")
    def test_embed_documents(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "aws-region-name"
        dims = 256

        embeddings = random(size=(2, dims))
        with patch(
            "src.core.embeddings.bedrock.BedrockEmbeddings._embedding_func"
        ) as mock_embedding_func:
            mock_embedding_func.return_value = random(size=dims)

            bedrock_embeddings = BedrockEmbeddings(
                model_id, aws_region_name, mock_client, dims=dims
            )

            self.assertEqual(
                len(bedrock_embeddings.embed_documents(["test-text", "test-text"])), len(embeddings)
            )
            self.assertEqual(
                len(bedrock_embeddings.embed_documents(["test-text", "test-text"])[0]),
                len(embeddings[0]),
            )

    @patch("boto3.client")
    def test_embed_query(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "aws-region-name"
        dims = 256

        embeddings = random(size=dims)
        with patch(
            "src.core.embeddings.bedrock.BedrockEmbeddings._embedding_func"
        ) as mock_embedding_func:
            mock_embedding_func.return_value = random(size=dims)
            bedrock_embeddings = BedrockEmbeddings(
                model_id, aws_region_name, mock_client, dims=dims
            )
            self.assertEqual(len(bedrock_embeddings.embed_query("test-text")), len(embeddings))

    @patch("boto3.client")
    def test_invalid_dimensions(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "aws-region-name"
        dims = 123

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client, dims=dims)

        with self.assertRaises(ValueError):
            bedrock_embeddings.embed_query("test-text")

    @patch("boto3.client")
    def test_invalid_model_id(self, mock_client):
        model_id = "amazon.titan-v2:0"
        aws_region_name = "aws-region-name"

        bedrock_embeddings = BedrockEmbeddings(model_id, aws_region_name, mock_client)
        with self.assertRaises(ValueError):
            bedrock_embeddings.embed_query("test-text")
