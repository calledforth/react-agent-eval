import pandas as pd
import random
import json
import os
from typing import List, Dict, Any, Tuple
from Agent import Agent
import re
from wikipedia_tool import get_wikipedia_content


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
        self.extracted_labels = []  # Add extraction of labels
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

        # Extract both claims and labels
        self.extracted_claims = self.dataset["claim"].tolist()
        self.extracted_labels = self.dataset["label"].tolist()
        self.length = len(self.extracted_claims)

        return f"Successfully loaded fever dataset of size {self.length} from {self.dataset_path}"

    def get_claims(self, num_claims: int = 5) -> List[Tuple[str, str]]:
        """
        Get a specified number of claims from the dataset.
        If num_claims is 0, returns all claims.
        If num_claims > 0, returns that many randomly selected claims.

        Args:
            num_claims (int): Number of claims to retrieve. 0 for all claims.

        Returns:
            List of tuples containing (claim, label)
        """
        if num_claims == 0:
            return list(zip(self.extracted_claims, self.extracted_labels))

        elif num_claims > self.length:
            raise ValueError(
                f"Requested number of claims exceeds dataset size ({self.length})"
            )

        # Generate random indices to select both claims and labels at the same positions
        indices = random.sample(range(self.length), num_claims)
        selected_claims = [self.extracted_claims[i] for i in indices]
        selected_labels = [self.extracted_labels[i] for i in indices]

        return list(zip(selected_claims, selected_labels))

    def eval_claims(self, claim_label_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the claims by running them through the agent and comparing to ground truth.

        Args:
            claim_label_pairs: List of tuples containing (claim, label) to evaluate

        Returns:
            Dictionary with evaluation results
        """
        agent = Agent()
        self.correct_verifications = 0
        self.claim_verification_pairs = []
        evaluation_progress = {
            "current_claim": 0,
            "total_claims": len(claim_label_pairs),
            "thinking_round": 0,
            "status": "",
        }

        # Mapping between FEVER dataset labels and agent output
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

        for index, (claim, label) in enumerate(claim_label_pairs):
            print(f"EVALUATING CLAIM : {index + 1}")
            print(f"CLAIM: {claim}")
            print(f"GROUND TRUTH LABEL: {label}\n")
            message = claim

            # Update progress information
            evaluation_progress["current_claim"] = index + 1
            evaluation_progress["status"] = "evaluating"
            evaluation_progress["claim_text"] = claim

            verification_result = None

            for num in range(1, 8):
                # Update thinking round
                evaluation_progress["thinking_round"] = num

                raw_response = agent.fever_chat(message)

                try:
                    raw_response = raw_response["choices"][0]["message"]["content"]
                    print("PARSING RESPONSE")
                    response = parse_json_from_response(raw_response)

                    if "thinking" in response.keys():
                        print(f"THINKING ROUND {num}: {response['thinking']}")

                        action = response["action"]
                        print(f"ACTION ROUND {num} : {action}\n")

                        # Handle different action types
                        if action.startswith("retrieve:"):
                            # Extract the entity to retrieve from Wikipedia
                            entity = action[len("retrieve:") :].strip()
                            print(f"Getting Wikipedia content for: {entity}...")
                            context = get_wikipedia_content(entity)
                            print("Wikipedia content retrieved\n")
                        elif action.startswith("search:"):
                            # Extract the search query
                            query = action[len("search:") :].strip()
                            print(f"Searching for: {query}...")
                            context = agent.answering_agent(query)
                            print("Search results retrieved\n")
                        else:
                            # Default to treating it as a Wikipedia retrieval
                            print("Getting Wikipedia content...")
                            context = get_wikipedia_content(action)
                            print("Wikipedia content retrieved\n")

                        message = f"{message}\n{response}\n{context}"
                        continue

                    elif "verification" in response.keys():
                        verification = response["verification"]
                        evidence = response.get("evidence", "")
                        print(f"\nVERIFICATION: {verification}")
                        print(f"EVIDENCE: {evidence}\n")

                        # Update progress information - completed verification
                        evaluation_progress["status"] = "completed"

                        if verification not in labels:
                            print(
                                f"❌ Couldn't map verification to a FEVER label: {verification}"
                            )
                            verification_result = False
                        else:
                            # Compare with ground truth
                            if verification.upper() == label.upper():
                                print("✅ VERIFICATION EVALUATION: CORRECT")
                                self.correct_verifications += 1
                                verification_result = True
                            else:
                                print(
                                    f"❌ VERIFICATION EVALUATION: INCORRECT (predicted {verification}, actual {label})"
                                )
                                verification_result = False

                        # Store the claim-verification pair
                        self.claim_verification_pairs.append(
                            {
                                "claim": claim,
                                "ground_truth": label,
                                "verification": verification,
                                "evidence": evidence,
                                "correct": verification_result,
                            }
                        )
                        break

                except Exception as e:
                    print(f"Error processing claim {index + 1}: {e}")
                    verification_result = False
                    # Update progress information - error
                    evaluation_progress["status"] = "error"
                    self.claim_verification_pairs.append(
                        {
                            "claim": claim,
                            "ground_truth": label,
                            "verification": "Error processing claim",
                            "evidence": str(e),
                            "correct": False,
                        }
                    )
                    break

            # If we went through all rounds without a verification
            if verification_result is None:
                print("❌ No verification produced after maximum rounds")
                # Update progress information - failed
                evaluation_progress["status"] = "failed"
                self.claim_verification_pairs.append(
                    {
                        "claim": claim,
                        "ground_truth": label,
                        "verification": "No verification produced",
                        "evidence": "",
                        "correct": False,
                    }
                )

        return {
            "correct_verifications": self.correct_verifications,
            "claim_verification_pairs": self.claim_verification_pairs,
            "evaluation_progress": evaluation_progress,
        }
