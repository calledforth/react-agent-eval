import json
import os
import random
from typing import List, Dict, Any
from Agent import Agent
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
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

    def eval_questions(self, questions: List[str]) -> Dict[str, Any]:
        """
        Evaluate the questions by printing them out.
        Modified to also store results for Streamlit display.
        """
        agent = Agent()
        for index, question in enumerate(questions):
            print(f"EVALUATING QUESTION : {index + 1}")
            print(f"QUESTION: {question}\n")
            message = question
            for num in range(1, 8):
                # print(f"Message {num}: {message}")
                raw_response = agent.hotpotqa_chat(message)

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

                    elif "answer" in response.keys():
                        answer = response["answer"]
                        print(f"\nANSWER: {answer}\n")

                        # Evaluate the answer
                        evaluation_result = agent.evaluation_agent(question, answer)
                        if evaluation_result == 1:
                            print("✅ ANSWER EVALUATION : VALID")
                            self.correct_answers += 1
                            # Store the successful question-answer pair
                            self.question_answer_pairs.append(
                                {"question": question, "answer": answer, "valid": True}
                            )
                            break
                        elif evaluation_result == 0:
                            print("❌ ANSWER EVALUATION: INVALID")
                            self.question_answer_pairs.append(
                                {"question": question, "answer": answer, "valid": False}
                            )
                            break

                except Exception as e:
                    print(f"Error processing question {index + 1}: {e}")

        return {
            "correct_answers": self.correct_answers,
            "question_answer_pairs": self.question_answer_pairs,
        }


def run_streamlit_app():
    # Set page to wide mode to use the full screen width
    st.set_page_config(layout="wide")

    st.title("HotpotQA Evaluation with Streamlit")

    # Input section at the top of the screen
    st.header("Configuration")

    # Create three columns for inputs
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # Input section for dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value="datasets\\hotpot_dev_fullwiki_v1.json",
            help="Path to the HotpotQA dataset JSON file",
        )

    with col2:
        # Input for number of questions
        num_questions = st.number_input(
            "Number of Questions",
            min_value=0,
            value=1,
            help="Choose how many questions to sample from the dataset (0 for all)",
        )

    with col3:
        # Run evaluation button (vertically centered)
        st.write("")  # Add some space
        run_button = st.button("Run Evaluation", use_container_width=True)

    # Horizontal line to separate inputs from results
    st.markdown("---")

    # Create two columns for logs and results side by side with equal width
    log_col, results_col = st.columns(2)

    # Logs column
    with log_col:
        st.header("Evaluation Logs")
        logs_area = st.empty()

    # Results column
    with results_col:
        st.header("Evaluation Results")
        # Container for metrics
        results_metrics = st.container()
        # Container for Q&A pairs with scrollable area
        qa_container = st.container()
        with qa_container:
            st.subheader("Question-Answer Pairs")
            # Create a container with fixed height and scrolling
            qa_scroll_container = st.container()

    # Run evaluation when button is clicked
    if run_button:
        try:
            # Set up logging to capture output
            orig_stdout = sys.stdout
            log_capture = StreamlitPrintCapture(logs_area)
            sys.stdout = log_capture

            # Run the evaluation
            with st.spinner("Evaluation in progress..."):
                hotpot_eval = HotpotQAEval(dataset_path)
                hotpot_eval.load_hotpotqa_dataset()
                questions_to_evaluate = hotpot_eval.get_questions(num_questions)
                result = hotpot_eval.eval_questions(questions_to_evaluate)

            # Restore stdout
            sys.stdout = orig_stdout

            # Display results
            with results_metrics:
                st.metric(
                    "Correct Answers",
                    result["correct_answers"],
                    f"{result['correct_answers']}/{len(questions_to_evaluate)} questions",
                )

            # Add CSS to create a scrollable container with fixed height
            st.markdown(
                """
                <style>
                .scrollable-container {
                    height: 400px;
                    overflow-y: auto;
                    border: 1px solid #e6e6e6;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #000000;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            # Generate markdown for QA pairs
            qa_text = ""
            for idx, qa_pair in enumerate(result["question_answer_pairs"]):
                status = "✅ VALID" if qa_pair["valid"] else "❌ INVALID"
                qa_text += f"Question {idx+1}: {qa_pair['question']}\n\n"
                qa_text += f"Answer: {qa_pair['answer']}\n\n"
                qa_text += f"Status: {status}\n\n"
                qa_text += "---\n\n"

            # Display the markdown in a scrollable div
            with qa_scroll_container:
                st.markdown(
                    f'<div class="scrollable-container">{qa_text}</div>',
                    unsafe_allow_html=True,
                )

            st.success("Evaluation completed!")

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Check if running as script or via streamlit
    if "streamlit" in sys.modules:
        run_streamlit_app()
    else:
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
