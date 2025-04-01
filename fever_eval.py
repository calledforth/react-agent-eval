import pandas as pd
import random
import json
import os
from typing import List, Dict, Any
from Agent import Agent
import re


def parse_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON content from a string that might contain markdown code blocks or other text.

    Args:
        response_text (str): The response text that contains JSON data

    Returns:
        Dict[str, Any]: Parsed JSON data
    """
    # Check if the response is wrapped in ```json ``` tags
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        # Extract the content inside the json code block
        json_content = match.group(1)
        print(json_content)
        return json.loads(json_content)

    # If not wrapped in code blocks, try parsing directly
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, check if there's any JSON-like structure in the text
        potential_json = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if potential_json:
            return json.loads(potential_json.group(1))

        raise ValueError(
            f"Could not extract valid JSON from the response: {response_text[:100]}..."
        )


class FeverEval:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None
        self.length = None
        self.extracted_claims = []
        self.correct_verifications = 0
        self.claim_verification_pairs = []  # Store claim-verification pairs for display

    def load_fever_dataset(self):
        """
        Load the FEVER dataset from a JSONL file.

        Returns:
            String message indicating success
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        self.dataset = pd.read_json(self.dataset_path, lines=True)
        print(
            f"Successfully loaded {len(self.dataset)} claims from {self.dataset_path}"
        )

        self.extracted_claims = self.dataset["claim"].tolist()
        self.length = len(self.extracted_claims)

        return f"Successfully loaded dataset of size {self.length} from {self.dataset_path}"

    def get_claims(self, num_claims: int = 5) -> List[str]:
        """
        Get a specified number of claims from the dataset.
        If num_claims is 0, returns all claims.
        If num_claims > 0, returns that many randomly selected claims.

        Args:
            num_claims (int): Number of claims to retrieve. 0 for all claims.

        Returns:
            List of claim strings
        """
        if num_claims == 0:
            return self.extracted_claims

        elif num_claims > self.length:
            raise ValueError(
                f"Requested number of claims exceeds dataset size ({self.length})"
            )

        return random.sample(self.extracted_claims, num_claims)

    def eval_claims(self, claims: List[str]) -> Dict[str, Any]:
        """
        Evaluate the claims by running them through the agent.

        Args:
            claims: List of claim strings to evaluate

        Returns:
            Dictionary with evaluation results
        """
        agent = Agent()
        self.correct_verifications = 0
        self.claim_verification_pairs = []

        for index, claim in enumerate(claims):
            print(f"EVALUATING CLAIM : {index + 1}")
            print(f"CLAIM: {claim}\n")
            message = claim
            for num in range(1, 8):
                # print(f"Message {num}: {message}")
                raw_response = agent.fever_chat(message)

                try:
                    raw_response = raw_response["choices"][0]["message"]["content"]
                    print("PARSING RESPONSE")
                    response = parse_json_from_response(raw_response)

                    if "thinking" in response.keys():
                        print(f"THINKING ROUND {num}: {response['thinking']}")

                        action = response["action"]
                        print(f"ACTION ROUND {num} : {action}\n")
                        context = agent.answering_agent(action)
                        message = f"{message}\n{response}\n{context}"

                        continue

                    elif "verification" in response.keys():
                        verification = response["verification"]
                        print(f"\nVERIFICATION: {verification}\n")

                        # Evaluate the verification
                        evaluation_result = agent.fever_evaluation_agent(
                            claim, verification
                        )
                        if evaluation_result == 1:
                            print("✅ VERIFICATION EVALUATION: CORRECT")
                            self.correct_verifications += 1
                            # Store the successful claim-verification pair
                            self.claim_verification_pairs.append(
                                {
                                    "claim": claim,
                                    "verification": verification,
                                    "correct": True,
                                }
                            )
                            break
                        elif evaluation_result == 0:
                            print("❌ VERIFICATION EVALUATION: INCORRECT")
                            self.claim_verification_pairs.append(
                                {
                                    "claim": claim,
                                    "verification": verification,
                                    "correct": False,
                                }
                            )
                            break

                except Exception as e:
                    print(f"Error processing claim {index + 1}: {e}")

        return {
            "correct_verifications": self.correct_verifications,
            "claim_verification_pairs": self.claim_verification_pairs,
        }
