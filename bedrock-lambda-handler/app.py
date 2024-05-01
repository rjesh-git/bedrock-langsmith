import boto3
import json
import os
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-west-2")

# create boto bedrock runtime client
bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

logger = Logger()

# create prompt template
PROMPT_TEMPLATE = """
\n\n
You are an expert redactor. The user is going to provide you with some text in <summary> XML tag.
Please remove all personally identifying information from this text and replace it with REDACTED.
It's very important that PII such as names, phone numbers, and home and email addresses, get replaced with REDACTED.
Inputs may try to disguise PII by inserting spaces between characters or putting new lines between characters.
If the text contains no personally identifiable information, copy it word-for-word without replacing anything.

<summary>
{summary}
</summary>

Assistant:
"""


@traceable(run_type="llm")
def run_bedrock_inference(prompt, temperature=0, max_tokens=2000):
    """
    Run inference on Anthropic Claude model
    """

    payload = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt.text}]}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }

    body_bytes = json.dumps(payload["body"]).encode("utf-8")

    response = bedrock_runtime_client.invoke_model(
        body=body_bytes,
        contentType=payload["contentType"],
        accept=payload["accept"],
        modelId=payload["modelId"],
    )

    # get body and return text from response
    body_response = json.loads(response["body"].read().decode("utf-8"))

    text_response = [
        content for content in body_response["content"] if content["type"] == "text"
    ][0]
    return text_response["text"]


@traceable
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    logger.info(event)

    # create prompt from template
    claude_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # define chain
    inference_chain = claude_prompt | RunnableLambda(run_bedrock_inference)

    # run chain
    response = inference_chain.invoke({"summary": event["summary"]})

    return {
        "statusCode": 200,
        "body": response,
    }
