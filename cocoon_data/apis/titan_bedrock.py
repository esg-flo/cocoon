import json
import os

import boto3
from botocore.exceptions import ClientError

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1"
)

class TitanClient:
    def __init__(self, model_name: str = None, aws_region: str = "us-east-1", normalize: bool = False) -> None:
        self.client = boto3.client(
            service_name="bedrock-runtime", region_name=aws_region
        )
        self.model_name = model_name
        self.model_ids = {
            "titan-g1": "amazon.titan-embed-text-v1",
            "titan-v2": "amazon.titan-embed-text-v2:0"
        }
        self.output_vector_size = 1024
        self.normalize = normalize
        
        if self.model_name not in self.model_ids:
            raise Exception("Invalid Titan Model Name. It should be either 'titan-g1' or 'titan-v2'")

    def create(self, message, *args, **kwargs):
        message = self.parse_input(message)
        response = self.invoke_model(message, *args, **kwargs)
        return self.parse_output(response)

    def parse_input(self, message):
        if self.model_name == "titan-g1":
            return {
                "inputText": message
            }
        elif self.model_name == "titan-v2":
            return {
                "inputText": message,
                "dimensions": self.output_vector_size,
                "normalize": self.normalize
            }

    def parse_output(self, response):
        message_text = response.get("embedding")
        return message_text

    def invoke_model(self, message, *args, **kwargs):
        try:
            response = self.client.invoke_model(
                modelId=self.model_ids[self.model_name],
                body=json.dumps(message)
            )

            result = json.loads(response.get("body").read())
            if len(result.get("embedding", [])) == 0:
                return None

            return result
        except ClientError as err:
            raise


if __name__ == "__main__":
    client = TitanClient()
    embedding = client.create("Hello, how are you?")
    print(embedding)