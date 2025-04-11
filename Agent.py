import os
import base64
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
from wikipedia_tool import get_wikipedia_content

load_dotenv()


class Agent:
    def __init__(self, model_name="gpt-4o"):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = model_name  # Can be "gpt-4o" or "o3-mini"
        self.api_key = os.getenv("AZURE_OPENAI_API")
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-12-01-preview",
        )

    def hotpotqa_chat_react(self, thoughts):
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

        completion_params = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": messages}],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 800
        else:
            completion_params["max_tokens"] = 800

        completion = self.client.chat.completions.create(**completion_params)

        print("RECEIVED RESPONSE FROM HOTPOTQA AGENT\n")
        return json.loads(completion.to_json())

    def hotpotqa_chat_direct(self, question):
        print("SENDING MESSAGE TO DIRECT HOTPOTQA AGENT")
        chat_prompt = """
        You are an intelligent agent capable of answering complex multi-hop questions. 
        Given a question, provide a direct and complete answer based on your knowledge.

        Your answer should:
        1. Be comprehensive and well-reasoned
        2. Include relevant context and supporting details
        3. Address all parts of the multi-hop question
        4. Be factually accurate

        Output your response in this JSON format:
        {
            "answer": "your detailed answer here"
        }
        """

        messages = chat_prompt + "\nQuestion: " + question

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": messages}],
                "max_completion_tokens": 100000,
            }
        else:
            completion_params = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": messages}],
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "stream": False,
                "max_tokens": 800,
            }

        completion = self.client.chat.completions.create(**completion_params)

        print("RECEIVED RESPONSE FROM DIRECT HOTPOTQA AGENT\n")
        return json.loads(completion.to_json())

    def evaluation_agent(self, question, answer):

        print("EVALUATION AGENT IS EVALUATING ANSWER")
        chat_prompt = """
            You are an intelligent agent capable of evaluating answers to questions. Your task is to determine if the provided answer is valid or invalid based on the question.

            If the answer answers the question, return 1. If the answer is incorrect or does not answer the question, return 0.

            You can take a lenient approach to the evaluation, such that answers which are not 100% correct but are close enough can be considered valid. For example, if the answer is a paraphrase of the solution or a close approximation, you can return 1. If the answer is completely off-topic or irrelevant, you can return 0.

            You can return 0 in cases when the answer is not relevant to the question or does not provide any useful information or returns factually incorrect information. For example, if the answer is a random fact that does not relate to the question, you can return 0.

            you can return only 1 or 0."""

        completion_params = {
            "model": self.deployment,
            "messages": [
                {
                    "role": "user",
                    "content": f"{chat_prompt} Question: {question}\nAnswer: {answer}",
                },
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 800
        else:
            completion_params["max_tokens"] = 800

        completion = self.client.chat.completions.create(**completion_params)

        print("EVALUATION COMPLETE\n")
        final = json.loads(completion.to_json())
        return int(final["choices"][0]["message"]["content"])

    def answering_agent(self, query):
        # Initialize Azure OpenAI Service client with key-based authentication
        print("SENDING MESSAGE TO ANSWERING AGENT FOR KNOWLEDGE RETRIEVAL")
        chat_prompt = """
Analyze the given query and provide detailed information based on the context."""

        completion_params = {
            "model": self.deployment,
            "messages": [
                {
                    "role": "user",
                    "content": f"{chat_prompt} Query: {query}",
                },
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 800
        else:
            completion_params["max_tokens"] = 800

        completion = self.client.chat.completions.create(**completion_params)

        print("KNOWLEDGE CONTEXT RECIEVED\n")
        final = json.loads(completion.to_json())
        return final["choices"][0]["message"]["content"]

    def fever_chat_react(self, claim):
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

        completion_params = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": messages}],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 800
        else:
            completion_params["max_tokens"] = 800

        completion = self.client.chat.completions.create(**completion_params)

        print("RECEIVED RESPONSE FROM FEVER AGENT\n")
        return json.loads(completion.to_json())

    def fever_chat_direct(self, claim):
        print("SENDING MESSAGE TO DIRECT FEVER AGENT")
        chat_prompt = """
        You are an intelligent fact-checking agent capable of verifying factual claims.
        Given a claim, verify its truthfulness based on your knowledge.

        Provide your verification in this JSON format:
        {
            "verification": "SUPPORTS/REFUTES/NOT ENOUGH INFO",
            "evidence": "Your explanation of why you chose this verification"
        }

        Important:
        - SUPPORTS: Use when you are confident the claim is true
        - REFUTES: Use when you are confident the claim is false
        - NOT ENOUGH INFO: Use when you cannot confidently verify or refute the claim

        Base your verification purely on your built-in knowledge and provide clear reasoning in the evidence field.
        """

        messages = chat_prompt + "\nClaim: " + claim

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": messages}],
                "max_completion_tokens": 100000,
            }
        else:
            completion_params = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": messages}],
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "stream": False,
                "max_tokens": 800,
            }

        completion = self.client.chat.completions.create(**completion_params)

        print("RECEIVED RESPONSE FROM DIRECT FEVER AGENT\n")
        return json.loads(completion.to_json())

    def alfworld_chat_react(self, task):
        print("SENDING MESSAGE TO ALFWORLD REACT AGENT")
        chat_prompt = """
        You are an intelligent agent in an interactive home environment, tasked with performing household tasks through a sequence of actions.
        
        You must follow a strict process consisting of:
        1. **Thinking** — reason about what information you currently have, what actions are available, and what to do next.
        2. **Action** — select the next action to perform based on your thinking.
        3. Take one step at a time and build upon previous observations.
        
        ### Your Workflow:
        
        You will work in a loop of:
        1. Thinking: Reflect on the task, what you've observed, and what needs to be done next.
        2. Action: Choose a specific action to perform.
        
        ### Available Actions:
        - look: Look around to observe the environment
        - move: Move to another location (e.g., "move to kitchen")
        - take: Take/pick up an object (e.g., "take apple")
        - place: Place an object somewhere (e.g., "place apple in fridge")
        - open: Open a container (e.g., "open fridge")
        - close: Close a container (e.g., "close fridge")
        - use: Use an appliance or cookware such as pans (e.g., "use microwave")
        - clean: Clean an object (e.g., "clean apple")
        - heat: Heat an object (e.g., "heat apple")
        - cool: Cool an object (e.g., "cool apple")
        
        ### Output Format:
        
        For intermediate steps, output a JSON object with this format:
        ```json
        {
            "thinking": "your reasoning here",
            "action": "one specific action to take"
        }
        ```
        
        For example:
        ```json
        {
            "thinking": "I need to find the apple first. Let me look around to see what's in the room.",
            "action": "look"
        }
        ```
        
        When you believe the task is complete, respond with:
        ```json
        {
            "success": true/false,
            "actions": ["list", "of", "all", "actions", "taken"],
            "reasoning": "explanation of how you solved the task"
        }
        ```
        
        Remember you are only allowed a maximum of 7 rounds of thinking and action.
        """

        messages = chat_prompt + "\n" + task

        completion_params = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": messages}],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 800
        else:
            completion_params["max_tokens"] = 800

        completion = self.client.chat.completions.create(**completion_params)

        print("RECEIVED RESPONSE FROM ALFWORLD REACT AGENT\n")
        return json.loads(completion.to_json())

    def alfworld_chat_direct(self, task):
        print("SENDING MESSAGE TO DIRECT ALFWORLD AGENT")
        chat_prompt = """

        You are an intelligent household agent operating in a simulated environment.  
You will be given a task and an environment description. Your goal is to reason through the steps and determine the best sequence of actions to complete the task.

First, think step by step using the "reasoning" section—describe what you need to do and why, referencing objects, locations, and environment context.  
Then, list the corresponding actions for each thought in the correct order.


Think carefully and make sure each action is justified based on the environment.
        
        Provide your response in this JSON format:
        {
            "actions": ["list", "of", "actions", "to", "take"],
            "reasoning": "your explanation of why these actions would work with respect to the current environment. Focus on explaining how each action contributes to solving the task based on the environment description"
        }
        
        Available actions include:
        - look: Look around to observe the environment
        - move: Move to another location (e.g., "move to kitchen")
        - take: Take/pick up an object (e.g., "take apple")
        - place: Place an object somewhere (e.g., "place apple in fridge")
        - open: Open a container (e.g., "open fridge")
        - close: Close a container (e.g., "close fridge")
        - use: Use an appliance or cookware such as pans (e.g., "use microwave")
        - clean: Clean an object (e.g., "clean apple")
        - heat: Heat an object (e.g., "heat apple")
        - cool: Cool an object (e.g., "cool apple")
        
        Make sure your actions are in the correct sequence, consider all details from the environment description, and be specific about the objects you interact with.
        """

        messages = chat_prompt + "\n" + task

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": messages}],
                "max_completion_tokens": 100000,
            }
        else:
            completion_params = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": messages}],
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "stream": False,
                "max_tokens": 800,
            }

        completion = self.client.chat.completions.create(**completion_params)

        print("RECEIVED RESPONSE FROM DIRECT ALFWORLD AGENT\n")
        return json.loads(completion.to_json())

    def alfworld_observation_agent(self, action):
        """
        Simulates the environment's response to an agent action.

        Args:
            action: The action taken by the agent

        Returns:
            A simulated observation based on the action
        """
        print("GENERATING SIMULATED OBSERVATION FOR:", action)
        chat_prompt = """
        You are simulating an interactive household environment. Given an action taken by an agent, 
        provide a realistic observation that might result from that action. 
        
        Be specific about what the agent would observe, including:
        - If looking: Describe visible objects, furniture, and layout
        - If taking an object: Whether it was successful, what the object feels like
        - If placing an object: Whether it fits, how it looks in its new location
        - If opening/closing: What's visible inside containers
        - If using appliances: Results of using them
        - If cleaning/heating/cooling: How the object changes
        
        Your observation should be concise but informative, helping the agent decide what to do next.
        """

        completion_params = {
            "model": self.deployment,
            "messages": [
                {
                    "role": "user",
                    "content": f"{chat_prompt}\nAction: {action}",
                },
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if self.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 200
        else:
            completion_params["max_tokens"] = 200

        completion = self.client.chat.completions.create(**completion_params)

        response = json.loads(completion.to_json())
        observation = response["choices"][0]["message"]["content"]
        print("OBSERVATION:", observation)
        return f"Observation: {observation}"
