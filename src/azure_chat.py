import openai
import os
from langchain_openai import AzureChatOpenAI

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.azure_endpoint = os.environ['AZURE_DEV_ENDPOINT']
openai.api_key = os.environ['AZURE_DEV_API_KEY']
model_version = os.environ['AZURE_DEV_MODEL_VERSION']

model = AzureChatOpenAI(
    deployment_name=model_version,
    temperature=0.0,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    azure_endpoint=openai.azure_endpoint,
    openai_api_key=openai.api_key,
)
