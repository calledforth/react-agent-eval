import json
import os
import random
from typing import List, Dict, Any
from Agent import Agent
import requests
from bs4 import BeautifulSoup
import re
import sys
from io import StringIO


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


# Redirect print statements to capture logs for streamlit
class StreamlitPrintCapture:
    def __init__(self, log_container):
        self.log_container = log_container
        self.buffer = StringIO()
        # Create a unique identifier for this instance
        self.instance_id = 1

    def write(self, text):

        self.buffer.write(text)
        self.log_container.text_area(
            "Log Output",
            self.buffer.getvalue(),
            height=400,
            key=f"log_output_{self.instance_id}",  # Use unique key with instance id
            help="Execution logs of the evaluation process",
        )
        self.instance_id += 1  # Increment instance id for next write
        return len(text)

    def flush(self):
        pass


class HotpotQAEval:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.length = None
        self.extracted_questions = []
        self.correct_answers = 0
        self.question_answer_pairs = []  # Store question-answer pairs for display
        self.react_question_answer_pairs = []
        self.direct_question_answer_pairs = []
        self.o3mini_question_answer_pairs = []
        self.react_correct_answers = 0
        self.direct_correct_answers = 0
        self.o3mini_correct_answers = 0

    def load_hotpotqa_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the HotpotQA dataset from a JSON file.

        Returns:
            List of question data entries
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Successfully loaded {len(data)} questions from {self.dataset_path}")
        self.extracted_questions = [dict["question"] for dict in data]

        self.length = len(self.extracted_questions)

        return (
            "Successfully loaded dataset of size "
            + str(self.length)
            + " from"
            + self.dataset_path
        )

    def get_questions(
        self, num_questions: int = 5, data: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a specified number of questions from the dataset.
        If num_questions is 0, returns all questions.
        If num_questions > 0, returns that many randomly selected questions.

        Args:
            num_questions (int): Number of questions to retrieve. 0 for all questions.

        Returns:
            List of question data entries
        """
        if num_questions == 0:
            return self.extracted_questions

        elif num_questions > self.length:
            raise ValueError(
                f"Requested number of questions exceeds dataset size ({self.length})"
            )

        return random.sample(self.extracted_questions, num_questions)

    def eval_questions(
        self, questions: List[str], use_react=True, use_gpt4o=True, use_o3mini=True
    ) -> Dict[str, Any]:
        """
        Evaluate the questions by printing them out.
        Modified to also store results for Streamlit display.

        Args:
            questions: List of questions to evaluate
            use_react: Whether to evaluate using the React agent
            use_gpt4o: Whether to evaluate using the GPT-4o direct agent
            use_o3mini: Whether to evaluate using the o3-mini direct agent

        Returns:
            Dictionary with evaluation results
        """
        agent_gpt4o = Agent("gpt-4o")
        agent_o3mini = Agent("o3-mini")
        evaluation_results = {
            "react_results": {"correct_answers": 0, "question_answer_pairs": []},
            "direct_results": {"correct_answers": 0, "question_answer_pairs": []},
            "o3mini_results": {"correct_answers": 0, "question_answer_pairs": []},
            "evaluation_progress": {
                "current_question": 0,
                "total_questions": len(questions),
                "current_agent": "",
                "thinking_round": 0,
                "status": "",
            },
        }

        for index, question in enumerate(questions):
            print(f"\nEVALUATING QUESTION {index + 1}")
            print(f"QUESTION: {question}\n")

            # Update progress information
            evaluation_results["evaluation_progress"]["current_question"] = index + 1
            evaluation_results["evaluation_progress"]["status"] = "evaluating"

            # Direct GPT-4o agent evaluation if selected
            if use_gpt4o:
                evaluation_results["evaluation_progress"][
                    "current_agent"
                ] = "Direct Agent (GPT-4o)"
                try:
                    print("DIRECT GPT-4O AGENT EVALUATION:")
                    raw_response = agent_gpt4o.hotpotqa_chat_direct(question)
                    direct_response = parse_json_from_response(
                        raw_response["choices"][0]["message"]["content"]
                    )

                    direct_answer = direct_response["answer"]
                    print(f"DIRECT GPT-4O ANSWER: {direct_answer}\n")

                    direct_evaluation = agent_gpt4o.evaluation_agent(
                        question, direct_answer
                    )
                    if direct_evaluation == 1:
                        print("✅ DIRECT GPT-4O ANSWER EVALUATION: VALID")
                        evaluation_results["direct_results"]["correct_answers"] += 1
                        evaluation_results["direct_results"][
                            "question_answer_pairs"
                        ].append(
                            {
                                "question": question,
                                "answer": direct_answer,
                                "valid": True,
                            }
                        )
                    else:
                        print("❌ DIRECT GPT-4O ANSWER EVALUATION: INVALID")
                        evaluation_results["direct_results"][
                            "question_answer_pairs"
                        ].append(
                            {
                                "question": question,
                                "answer": direct_answer,
                                "valid": False,
                            }
                        )
                except Exception as e:
                    print(f"Error in direct GPT-4o agent: {e}")
                    evaluation_results["direct_results"][
                        "question_answer_pairs"
                    ].append(
                        {
                            "question": question,
                            "answer": f"Error: {str(e)}",
                            "valid": False,
                        }
                    )

            # Direct o3-mini agent evaluation if selected
            if use_o3mini:
                evaluation_results["evaluation_progress"][
                    "current_agent"
                ] = "Direct Agent (o3-mini)"
                try:
                    print("DIRECT O3-MINI AGENT EVALUATION:")
                    raw_response = agent_o3mini.hotpotqa_chat_direct(question)
                    o3mini_response = parse_json_from_response(
                        raw_response["choices"][0]["message"]["content"]
                    )

                    o3mini_answer = o3mini_response["answer"]
                    print(f"DIRECT O3-MINI ANSWER: {o3mini_answer}\n")

                    # Use the GPT-4o agent for evaluation to ensure fair comparison
                    o3mini_evaluation = agent_gpt4o.evaluation_agent(
                        question, o3mini_answer
                    )
                    if o3mini_evaluation == 1:
                        print("✅ DIRECT O3-MINI ANSWER EVALUATION: VALID")
                        evaluation_results["o3mini_results"]["correct_answers"] += 1
                        evaluation_results["o3mini_results"][
                            "question_answer_pairs"
                        ].append(
                            {
                                "question": question,
                                "answer": o3mini_answer,
                                "valid": True,
                            }
                        )
                    else:
                        print("❌ DIRECT O3-MINI ANSWER EVALUATION: INVALID")
                        evaluation_results["o3mini_results"][
                            "question_answer_pairs"
                        ].append(
                            {
                                "question": question,
                                "answer": o3mini_answer,
                                "valid": False,
                            }
                        )
                except Exception as e:
                    print(f"Error in direct o3-mini agent: {e}")
                    evaluation_results["o3mini_results"][
                        "question_answer_pairs"
                    ].append(
                        {
                            "question": question,
                            "answer": f"Error: {str(e)}",
                            "valid": False,
                        }
                    )

            # React agent evaluation (using GPT-4o) if selected
            if use_react:
                evaluation_results["evaluation_progress"][
                    "current_agent"
                ] = "React Agent"
                print("\nREACT AGENT EVALUATION:")
                message = question
                react_verification_result = None

                for num in range(1, 8):
                    # Update thinking round for UI feedback
                    evaluation_results["evaluation_progress"]["thinking_round"] = num

                    try:
                        raw_response = agent_gpt4o.hotpotqa_chat_react(message)
                        raw_response = raw_response["choices"][0]["message"]["content"]
                        print("PARSING RESPONSE")
                        response = parse_json_from_response(raw_response)

                        if "thinking" in response.keys():
                            print(f"THINKING ROUND {num}: {response['thinking']}")
                            action = response["action"]
                            print(f"ACTION ROUND {num} : {action}\n")
                            context = agent_gpt4o.answering_agent(action)
                            message = f"{message}\n{response}\n{context}"
                            continue

                        elif "answer" in response.keys():
                            answer = response["answer"]
                            print(f"\nANSWER: {answer}\n")

                            # Evaluate the answer
                            evaluation_result = agent_gpt4o.evaluation_agent(
                                question, answer
                            )
                            if evaluation_result == 1:
                                print("✅ ANSWER EVALUATION : VALID")
                                evaluation_results["react_results"][
                                    "correct_answers"
                                ] += 1
                                evaluation_results["react_results"][
                                    "question_answer_pairs"
                                ].append(
                                    {
                                        "question": question,
                                        "answer": answer,
                                        "valid": True,
                                    }
                                )
                            else:
                                print("❌ ANSWER EVALUATION: INVALID")
                                evaluation_results["react_results"][
                                    "question_answer_pairs"
                                ].append(
                                    {
                                        "question": question,
                                        "answer": answer,
                                        "valid": False,
                                    }
                                )

                            # Update progress information - completed
                            evaluation_results["evaluation_progress"][
                                "status"
                            ] = "completed"
                            react_verification_result = True
                            break

                    except Exception as e:
                        print(
                            f"Error processing question with React agent {index + 1}: {e}"
                        )
                        evaluation_results["react_results"][
                            "question_answer_pairs"
                        ].append(
                            {
                                "question": question,
                                "answer": f"Error: {str(e)}",
                                "valid": False,
                            }
                        )
                        evaluation_results["evaluation_progress"]["status"] = "error"
                        react_verification_result = False
                        break

                # If we went through all rounds without getting an answer
                if react_verification_result is None:
                    print("❌ No answer produced after maximum rounds with React agent")
                    evaluation_results["evaluation_progress"]["status"] = "failed"
                    evaluation_results["react_results"]["question_answer_pairs"].append(
                        {
                            "question": question,
                            "answer": "No answer produced after maximum rounds",
                            "valid": False,
                        }
                    )

        return evaluation_results


if __name__ == "__main__":
    # Original command-line execution path
    dataset_path = "datasets\\hotpot_dev_fullwiki_v1.json"
    print("ENTER NUMBER OF QUESTIONS TO RETRIEVE (0 FOR ALL):")
    num_questions = int(input())

    try:
        print(f"Attempting to load dataset from: {dataset_path}")
        hotpot_eval = HotpotQAEval(dataset_path)
        hotpot_eval.load_hotpotqa_dataset()
        print("Obtain questions to evaluate")
        questions_to_evaluate = hotpot_eval.get_questions(num_questions)
        print("starting evaluation")
        result = hotpot_eval.eval_questions(questions_to_evaluate)
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
