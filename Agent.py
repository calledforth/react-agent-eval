import os
import base64
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
from wikipedia_tool import get_wikipedia_content

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

Make sure your final answer is complete and well-drafted response to the question, with proper context and information.

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

If the answer answers the question, return 1. If the answer is incorrect or does not answer the question, return 0.

You can take a lenient approach to the evaluation, such that answers which are not 100% correct but are close enough can be considered valid. For example, if the answer is a paraphrase of the solution or a close approximation, you can return 1. If the answer is completely off-topic or irrelevant, you can return 0.

You can return 0 in cases when the answer is not relevant to the question or does not provide any useful information or returns factually incorrect information. For example, if the answer is a random fact that does not relate to the question, you can return 0.

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

    def fever_chat(self, claim):
        # Initialize Azure OpenAI Service client with key-based authentication
        print("SENDING MESSAGE TO FEVER AGENT")
        chat_prompt = """
You are an intelligent fact-checking agent capable of verifying factual claims by interacting with a simulated knowledge environment. 

You must follow a strict process consisting of:
1. **Thinking** — you will reason about what information you currently have and decide what to do next.
2. **Action** — you will select and execute actions to retrieve new information or give the final verification.
3. Every step you can use only one action and think on one substep at a time and not attempt to verify the entire claim at once.
---

### Your Workflow:

You will work in a loop of:
1. Thinking: Reflect on what you know and what you need to verify.
2. Action: Action to be performed.   
---

Available Actions:
- retrieve: Request information about a specific topic, entity, or fact from Wikipedia. Use this for direct lookups about entities, events, or topics that have Wikipedia articles, one important condition for this action is that the entity for this action must be a single entity or topic and no combination of entities or topics.
Eg : "retrieve: Elon Musk" or "retrieve: Paris" or "retrieve: World War II" or "retrieve: Dan Brown".
- search: Request information about a query that may not be directly available on Wikipedia or requires general knowledge. Use this for complex questions or when you need information beyond Wikipedia. Use this action for relationships between entities or topics, or when the entity for this action is a combination of entities or topics.

### Output Format:

You must **always output a JSON object** with the following format:

For intermediate steps:
```json
{
    "thinking": "describe your reasoning here.",
    "action": "retrieve: [specific entity or topic]"
}

OR

```json
{
    "thinking": "describe your reasoning here.",
    "action": "search: [question or query]"
}

For example:
{
    "thinking": "I need to find information about Elon Musk to verify his birth year.",
    "action": "retrieve: Elon Musk"
}

OR

{
    "thinking": "I need to understand what the typical climate is in Mediterranean regions.",
    "action": "search: What is the typical climate in Mediterranean regions?"
}

Once final verification is reached reply with the below format only:
{
    "verification": "your final verification here, stating whether the claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO",
    "evidence": "evidence supporting the verification"
}

Important: When providing your final verification, please use one of these three labels exactly:
- SUPPORTS (if the claim is supported by the evidence)
- REFUTES (if the evidence contradicts the claim)
- NOT ENOUGH INFO (if there's insufficient evidence to determine

Remember you are only allowed a maximum of 7 rounds of thinking and action.
So don't waste your turns on unnecessary actions and thinking, use your turns wisely.
"""

        messages = chat_prompt + claim

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

        print("RECEIVED RESPONSE FROM FEVER AGENT\n")
        return json.loads(completion.to_json())
