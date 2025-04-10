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
        self.length = None
        self.claims = []
        self.react_claim_verification_pairs = []
        self.direct_claim_verification_pairs = []
        self.o3mini_claim_verification_pairs = []
        self.react_correct_verifications = 0
        self.direct_correct_verifications = 0
        self.o3mini_correct_verifications = 0
        self.evaluation_progress = {
            "current_claim": 0,
            "total_claims": 0,
            "current_agent": "",
            "thinking_round": 0,
            "status": "",
        }

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

    def eval_claims(self, claims_with_labels: List[Tuple[str, str]], use_react=True, use_gpt4o=True, use_o3mini=True) -> Dict[str, Any]:
        """
        Evaluate the claims by running them through the agent and comparing to expected outcomes.

        Args:
            claims_with_labels: List of tuples containing claim text and ground truth label
            use_react: Whether to evaluate using the React agent
            use_gpt4o: Whether to evaluate using the GPT-4o direct agent
            use_o3mini: Whether to evaluate using the o3-mini direct agent

        Returns:
            Dictionary with evaluation results
        """
        agent_gpt4o = Agent("gpt-4o")
        agent_o3mini = Agent("o3-mini")
        self.evaluation_progress = {
            "current_claim": 0,
            "total_claims": len(claims_with_labels),
            "current_agent": "",
            "thinking_round": 0,
            "status": "",
        }

        evaluation_results = {
            "react_results": {
                "correct_verifications": 0,
                "claim_verification_pairs": [],
            },
            "direct_results": {
                "correct_verifications": 0,
                "claim_verification_pairs": [],
            },
            "o3mini_results": {
                "correct_verifications": 0,
                "claim_verification_pairs": [],
            },
            "evaluation_progress": self.evaluation_progress,
        }

        for index, (claim, label) in enumerate(claims_with_labels):
            print(f"\nEVALUATING CLAIM {index + 1}")
            print(f"CLAIM: {claim}")
            print(f"GROUND TRUTH: {label}\n")

            # Update progress information
            self.evaluation_progress["current_claim"] = index + 1
            self.evaluation_progress["status"] = "evaluating"
            evaluation_results["evaluation_progress"] = self.evaluation_progress

            # Direct GPT-4o agent evaluation if selected
            if use_gpt4o:
                self.evaluation_progress["current_agent"] = "Direct Agent (GPT-4o)"
                try:
                    print("DIRECT GPT-4O AGENT EVALUATION:")
                    raw_response = agent_gpt4o.fever_chat_direct(claim)
                    direct_response = parse_json_from_response(
                        raw_response["choices"][0]["message"]["content"]
                    )

                    direct_verification = direct_response["verification"]
                    direct_evidence = direct_response.get("evidence", "")
                    print(f"DIRECT GPT-4O VERIFICATION: {direct_verification}")
                    print(f"DIRECT GPT-4O EVIDENCE: {direct_evidence}\n")

                    # Check if the verification matches the ground truth
                    is_correct = (
                        direct_verification.strip().upper() == label.strip().upper()
                    )
                    if is_correct:
                        print("✅ DIRECT GPT-4O VERIFICATION: CORRECT")
                        evaluation_results["direct_results"]["correct_verifications"] += 1
                    else:
                        print("❌ DIRECT GPT-4O VERIFICATION: INCORRECT")

                    evaluation_results["direct_results"]["claim_verification_pairs"].append(
                        {
                            "claim": claim,
                            "ground_truth": label,
                            "verification": direct_verification,
                            "evidence": direct_evidence,
                            "correct": is_correct,
                        }
                    )
                except Exception as e:
                    print(f"Error in direct GPT-4o agent: {e}")
                    evaluation_results["direct_results"]["claim_verification_pairs"].append(
                        {
                            "claim": claim,
                            "ground_truth": label,
                            "verification": "ERROR",
                            "evidence": str(e),
                            "correct": False,
                        }
                    )

            # Direct o3-mini agent evaluation if selected
            if use_o3mini:
                self.evaluation_progress["current_agent"] = "Direct Agent (o3-mini)"
                try:
                    print("DIRECT O3-MINI AGENT EVALUATION:")
                    raw_response = agent_o3mini.fever_chat_direct(claim)
                    o3mini_response = parse_json_from_response(
                        raw_response["choices"][0]["message"]["content"]
                    )

                    o3mini_verification = o3mini_response["verification"]
                    o3mini_evidence = o3mini_response.get("evidence", "")
                    print(f"DIRECT O3-MINI VERIFICATION: {o3mini_verification}")
                    print(f"DIRECT O3-MINI EVIDENCE: {o3mini_evidence}\n")

                    # Check if the verification matches the ground truth
                    is_correct = (
                        o3mini_verification.strip().upper() == label.strip().upper()
                    )
                    if is_correct:
                        print("✅ DIRECT O3-MINI VERIFICATION: CORRECT")
                        evaluation_results["o3mini_results"]["correct_verifications"] += 1
                    else:
                        print("❌ DIRECT O3-MINI VERIFICATION: INCORRECT")

                    evaluation_results["o3mini_results"]["claim_verification_pairs"].append(
                        {
                            "claim": claim,
                            "ground_truth": label,
                            "verification": o3mini_verification,
                            "evidence": o3mini_evidence,
                            "correct": is_correct,
                        }
                    )
                except Exception as e:
                    print(f"Error in direct o3-mini agent: {e}")
                    evaluation_results["o3mini_results"]["claim_verification_pairs"].append(
                        {
                            "claim": claim,
                            "ground_truth": label,
                            "verification": "ERROR",
                            "evidence": str(e),
                            "correct": False,
                        }
                    )

            # React agent evaluation (using GPT-4o) if selected
            if use_react:
                self.evaluation_progress["current_agent"] = "React Agent"
                print("\nREACT AGENT EVALUATION:")
                message = claim
                react_verification_result = None

                for num in range(1, 8):
                    # Update thinking round for UI feedback
                    self.evaluation_progress["thinking_round"] = num

                    try:
                        raw_response = agent_gpt4o.fever_chat_react(message)
                        raw_response = raw_response["choices"][0]["message"]["content"]
                        print("PARSING RESPONSE")
                        response = parse_json_from_response(raw_response)

                        if "thinking" in response.keys():
                            print(f"THINKING ROUND {num}: {response['thinking']}")
                            action = response["action"]
                            print(f"ACTION ROUND {num} : {action}\n")

                            # Check if action starts with "retrieve:" or "search:"
                            if action.startswith("retrieve:"):
                                entity = action.split("retrieve:")[1].strip()
                                print(f"RETRIEVING FROM WIKIPEDIA: {entity}")
                                wiki_content = get_wikipedia_content(entity)
                                message = f"{message}\n{response}\nWikipedia content about {entity}:\n{wiki_content}"
                            elif action.startswith("search:"):
                                query = action.split("search:")[1].strip()
                                print(f"SEARCHING: {query}")
                                search_result = agent_gpt4o.answering_agent(query)
                                message = f"{message}\n{response}\nSearch result for {query}:\n{search_result}"
                            else:
                                # Handle invalid action format
                                print(f"INVALID ACTION FORMAT: {action}")
                                message = f"{message}\n{response}\nError: Invalid action format. Please use 'retrieve:' or 'search:'."
                            continue

                        elif "verification" in response.keys():
                            verification = response["verification"]
                            evidence = response.get("evidence", "")
                            print(f"\nVERIFICATION: {verification}")
                            print(f"EVIDENCE: {evidence}\n")

                            # Check if the verification matches the ground truth
                            is_correct = (
                                verification.strip().upper() == label.strip().upper()
                            )
                            if is_correct:
                                print("✅ REACT VERIFICATION: CORRECT")
                                evaluation_results["react_results"][
                                    "correct_verifications"
                                ] += 1
                            else:
                                print("❌ REACT VERIFICATION: INCORRECT")

                            evaluation_results["react_results"][
                                "claim_verification_pairs"
                            ].append(
                                {
                                    "claim": claim,
                                    "ground_truth": label,
                                    "verification": verification,
                                    "evidence": evidence,
                                    "correct": is_correct,
                                }
                            )

                            # Update progress information - completed
                            self.evaluation_progress["status"] = "completed"
                            evaluation_results["evaluation_progress"] = (
                                self.evaluation_progress
                            )
                            react_verification_result = True
                            break

                    except Exception as e:
                        print(f"Error processing claim {index + 1} with React agent: {e}")
                        evaluation_results["react_results"][
                            "claim_verification_pairs"
                        ].append(
                            {
                                "claim": claim,
                                "ground_truth": label,
                                "verification": "ERROR",
                                "evidence": str(e),
                                "correct": False,
                            }
                        )
                        self.evaluation_progress["status"] = "error"
                        evaluation_results["evaluation_progress"] = self.evaluation_progress
                        react_verification_result = False
                        break

                # If we went through all rounds without getting a verification
                if react_verification_result is None:
                    print("❌ No verification produced after maximum rounds")
                    self.evaluation_progress["status"] = "failed"
                    evaluation_results["evaluation_progress"] = self.evaluation_progress
                    evaluation_results["react_results"]["claim_verification_pairs"].append(
                        {
                            "claim": claim,
                            "ground_truth": label,
                            "verification": "No verification produced after maximum rounds",
                            "evidence": "",
                            "correct": False,
                        }
                    )

        return evaluation_results
