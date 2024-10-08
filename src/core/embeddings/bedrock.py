"""
Amazon Embedding Models
- V1: Titan Embeddings G1
- V2: Titan Text Embeddings V2
"""

import json
from typing import Any, Dict, List, Optional

import boto3
from core.embeddings.base import Embeddings


class BedrockEmbeddings(Embeddings):
    def __init__(
        self,
        model_id: str,
        aws_region_name: str,
        client: Optional[Any] = None,
        normalize: Optional[bool] = False,
        dims: Optional[int] = None,
    ):
        self.client = (
            client if client else boto3.client("bedrock-runtime", region_name=aws_region_name)
        )
        self._model_id = model_id
        self.normalize = normalize
        self.dims = dims

    @property
    def dimensions(self):
        return self.dims

    @property
    def model_id(self):
        return self._model_id

    def _embedding_func(self, text: str) -> Dict[str, str]:
        provider = self.model_id.split(".")[0]
        input_body = dict()

        if provider == "amazon":
            if self.model_id.endswith("v2:0"):
                if not self.dims:
                    self.dims = 1024

                if self.dims not in [1024, 512, 256]:
                    raise ValueError(
                        "Invalid dimensions value. Valid values are 1024 (default), 512, 256."
                    )

                input_body["dimensions"] = self.dims
                input_body["normalize"] = self.normalize or False

            input_body["inputText"] = text
        else:
            raise ValueError(f"Invalid model id: {self.model_id}")

        body = json.dumps(input_body)
        print(body)
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            if provider == "amazon":
                return response_body.get("embedding")

        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = self._embedding_func(text)
            results.append(response)

        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embedding_func(text)
