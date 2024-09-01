import json
import os

import boto3
from botocore.exceptions import ClientError

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0"
)


class Client:
    def __init__(self, aws_region: str = "us-east-1", normalize: bool = True) -> None:
        self.client = boto3.client(
            service_name="bedrock-runtime", region_name=aws_region
        )
        self.output_vector_size = 1024
        self.normalize = normalize

    def create(self, message, *args, **kwargs):
        message = self.parse_input(message)
        response = self.invoke_model(message, *args, **kwargs)
        return self.parse_output(response)

    def parse_input(self, message):
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
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(message)
            )

            result = json.loads(response.get("body").read())
            if len(result.get("embedding", [])) == 0:
                return None

            return result
        except ClientError as err:
            raise


if __name__ == "__main__":
    client = Client()
    embedding = client.create("Hello, how are you?")
    print(embedding)