import os
import base64
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

load_dotenv()


class Agent:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = "gpt-4o"
        self.api_key = os.getenv("AZURE_OPENAI_API")
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-05-01-preview",
        )

    def hotpotqa_chat(self, thoughts):
        # Initialize Azure OpenAI Service client with key-based authentication
        print("SENDING MESSAGE TO HOTPOTQA AGENT")
        chat_prompt = """
You are an intelligent agent capable of solving complex multi-hop questions by interacting with a simulated knowledge environment. 

You must follow a strict process consisting of:
1. **Thinking** — you will reason about what information you currently have and decide what to do next.
2. **Action** — you will select and execute actions to retrieve new information or give the final answer.
3. Every step you can use only one action and think on one substep at a time and not attempt to solve the entire question at once.
---

### Your Workflow:

You will work in a loop of:
1. Thinking: Reflect on what you know and what you need to find.
2. Action: Action to be performed.   
---

Available Actions:
- knowledge_search: Request an external agent to retrieve knowledge related to your question. This must include the search item and the lookup keyword.


### Output Format:

You must **always output a JSON object** with the following format:

For intermediate steps:
```json
{
    "thinking": "describe your reasoning here.",
    "action": "A simple string describing the search and lookup you would "
}

{
    "thinking": "I need to find out which country Dubrovnik is located in.",
    "action": "which country is Dubrovnik located in?"
}

Once final answer is reached reply with the below format only
{
    "answer": "your final answer here"
}

Remember you are only allowed a maximum of 7 rounds of thinking and action.
So don't waste your turns on unnecessary actions and thinking, use your turns wisely.

        """

        messages = chat_prompt + thoughts

        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": messages}],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )

        print("RECEIVED RESPONSE FROM HOTPOTQA AGENT\n")
        return json.loads(completion.to_json())

    def evaluation_agent(self, question, answer):

        print("EVALUATION AGENT IS EVALUATING ANSWER")
        chat_prompt = """
You are an intelligent agent capable of evaluating answers to questions. Your task is to determine if the provided answer is valid or invalid based on the question.

If the answer accurately answers the question, return 1. If the answer is incorrect or does not answer the question, return 0.

you can return only 1 or 0."""

        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "user",
                    "content": f"{chat_prompt} Question: {question}\nAnswer: {answer}",
                },
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        print("EVALUATION COMPLETE\n")
        final = json.loads(completion.to_json())
        return int(final["choices"][0]["message"]["content"])

    def answering_agent(self, query):
        # Initialize Azure OpenAI Service client with key-based authentication
        print("SENDING MESSAGE TO ANSWERING AGENT FOR KNOWLEDGE RETRIEVAL")
        chat_prompt = """
Keep your responses crisp and reply to the answer based on the query provided."""

        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "user",
                    "content": f"{chat_prompt} Query: {query}",
                },
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        print("KNOWLEDGE CONTEXT RECIEVED\n")
        final = json.loads(completion.to_json())
        return final["choices"][0]["message"]["content"]
