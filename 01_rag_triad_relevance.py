import os
from typing import Optional, Union

from dotenv import load_dotenv
from trulens_eval import OpenAI, Bedrock
from trulens_eval.feedback import AzureOpenAI

from providers.bedrock_claude_provider import BedrockClaude

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def get_relevance(provider: Union[OpenAI, AzureOpenAI,Bedrock,BedrockClaude], prompt, response):
    response = provider.relevance(
        prompt=prompt,
        response=response
    )

    return response


if __name__ == "__main__":

    prompt = "Please answer question based on provided context only " \
             "<context> " \
             "Taj Mahal is the name of Rajib's house. It is situated on the banks of river Howrah " \
             "<question> Where is Taj Mahal?  "
    response = "I am very confident that Taj Mahal is in America."

    openai_provider = OpenAI()
    response_openai = get_relevance(openai_provider, prompt, response)
    print("Open AI Response ", response_openai)

    aws_bedrock_provider = Bedrock(model_id="amazon.titan-text-lite-v1")
    response_bedrock = get_relevance(aws_bedrock_provider, prompt, response)
    print("Bedrock Titan Lite Response ", response_bedrock)

    aws_bedrock_claude_provider = BedrockClaude(model_id="anthropic.claude-v2:1")
    response_claude = get_relevance(aws_bedrock_claude_provider, prompt, response)
    print("Bedrock Claude Response ", response_claude)
