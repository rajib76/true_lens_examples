import logging
from typing import Dict, Optional, Sequence

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.endpoint import BedrockEndpoint
from trulens_eval.utils.generated import re_0_10_rating

logger = logging.getLogger(__name__)

system_prompt= f"""
Human:
You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT.
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

A few additional scoring guidelines:

- Long RESPONSES should score equally well as short RESPONSES.

- Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the most RELEVANT.

- RESPONSE must be relevant to the entire PROMPT to get a score of 10.

- RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.

- RESPONSE that is RELEVANT to none of the PROMPT should get a score of 0.

- RESPONSE that is RELEVANT to some of the PROMPT should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

- RESPONSE that is RELEVANT to most of the PROMPT should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- RESPONSE that is RELEVANT to the entire PROMPT should get a score of 9 or 10.

- RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 10.

- RESPONSE that confidently FALSE should get a score of 0.

- RESPONSE that is only seemingly RELEVANT should get a score of 0.

- Never elaborate.

PROMPT: {{prompt}}

RESPONSE: {{response}}

Assistant: """

system_prompt_v1 = f"""
Human:
You are a helpful RELEVANCE scorer assistant. You calculate the relevance score a given <RESPONSE> to the given <PROMPT>.

Here are some rules that you must follow while scoring the relevance

<RULES>
<RULE>
ALWAYS respond with a number between 0 and 10 as the relevance score. 0 is the least RELEVANT and 10 is the most RELEVANT.
</RULE>
<RULE>
RESPONSES should get a score of 10 ONLY when the entire RESPONSE is RELEVANT to the PROMPT.
</RULE>
<RULE>
RESPONSES should get a score of 0 if it is completely IRRELEVANT to the PROMPT.
</RULE>
<RULE>
RESPONSES which are somewhat RELEVANT should get a score of 2,3, or 4. HIGHER score indicates more relevance.
</RULE>
<RULE>
RESPONSES which are mostly RELEVANT should get a score of 5,6,7, or 8. HIGHER score indicates more relevance.
</RULE>
<RULE>
RESPONSES which are RELEVANT to the entire PROMPT should get a score of 9 ot 10.
</RULE>
<RULE>
RESPONSES which refuses to answer the PROMPT should get a score of 0.
</RULE>
</RULES>

<PROMPT>
{{prompt}} 
</PROMPT>

<RESPONSE>
{{response}}
</RESPONSE>

*REMEMBER* to respond only with the relevance score. PLEASE do not add anything else to the repsonse.

Assistant:
"""


# system_prompt_v1 = "Human:You are a helpful grading assistant. You will provide a relevance score between the given " \
#                 "<RESPONSE> and the given " \
#                 "<PROMPT>. Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most " \
#                 "relevant. Below is the PROMPT and the RESPONSE that you need to grade" \
#                 "\n <PROMPT> \n {prompt} \n <RESPONSE>\n {response}\n Assistant: " \
#                 " "


class BedrockClaude(LLMProvider):
    # LLMProvider requirement which we do not use:
    model_engine: str = "Bedrock"

    model_id: str
    endpoint: BedrockEndpoint

    def __init__(
            self, *args, model_id: str = "anthropic.claude-v2:1", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of AWS Feedback Functions.

        Parameters:

        - model_id (str, optional): The specific model id. Defaults to
          "amazon.titan-tg1-large".

        - All other args/kwargs passed to BedrockEndpoint and subsequently
          to boto3 client constructor.
        """

        # SingletonPerName: return singleton unless client provided
        if hasattr(self, "model_id") and "client" not in kwargs:
            return

        # Pass kwargs to Endpoint. Self has additional ones.
        self_kwargs = dict()
        self_kwargs.update(**kwargs)

        self_kwargs['model_id'] = model_id

        self_kwargs['endpoint'] = BedrockEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # LLMProvider requirement
    def _create_chat_completion(
            self,
            prompt: Optional[str] = None,
            messages: Optional[Sequence[Dict]] = None,
            **kwargs
    ) -> str:
        assert self.endpoint is not None
        assert prompt is not None, "Bedrock can only operate on `prompt`, not `messages`."

        import json

        body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 512, "temperature": 0.2, "top_p": 0.9})

        modelId = self.model_id

        response = self.endpoint.client.invoke_model(body=body, modelId=modelId)
        response_json = json.loads(response['body'].read().decode('utf-8'))
        print(response_json)
        response_body = response_json["completion"]
        return response_body

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses chat completion model. A function that completes a
        template to check the relevance of the response to a prompt.

        **Usage:**
        ```python
        feedback = Feedback(provider.relevance).on_input_output()
        ```

        The `on_input_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Usage on RAG Contexts:

        ```python
        feedback = Feedback(provider.relevance).on_input().on(
            TruLlama.select_source_nodes().node.text # See note below
        ).aggregate(np.mean)
        ```

        The `on(...)` selector can be changed. See [Feedback Function Guide :
        Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

        Parameters:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        prompt = system_prompt.format(prompt=prompt, response=response)
        return re_0_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    prompt=system_prompt.format(prompt=prompt, response=response)
                    # prompt=str.format(
                    #     system_prompt, prompt=prompt, response=response
                    # )
                )
            )
        ) / 10.0
