import streamlit as st
import sys
from hotpotqa_eval import HotpotQAEval, StreamlitPrintCapture
from fever_eval import FeverEval


def run_streamlit_app():
    # Set page to wide mode to use the full screen width
    st.set_page_config(layout="wide", page_title="Evaluation Dashboard")

    st.title("Evaluation Dashboard")

    # Create tabs for HotpotQA and FEVER
    tab1, tab2 = st.tabs(["HotpotQA Evaluation", "FEVER Evaluation"])

    with tab1:
        run_hotpotqa_evaluation()

    with tab2:
        run_fever_evaluation()


def run_hotpotqa_evaluation():
    st.header("HotpotQA Evaluation")

    # Input section for configuration
    st.subheader("Configuration")

    # Create three columns for inputs
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # Input section for dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value="datasets\\hotpot_dev_fullwiki_v1.json",
            help="Path to the HotpotQA dataset JSON file",
            key="hotpotqa_dataset_path",
        )

    with col2:
        # Input for number of questions
        num_questions = st.number_input(
            "Number of Questions",
            min_value=0,
            value=1,
            help="Choose how many questions to sample from the dataset (0 for all)",
            key="hotpotqa_num_questions",
        )

    with col3:
        # Run evaluation button (vertically centered)
        st.write("")  # Add some space
        run_button = st.button(
            "Run Evaluation", use_container_width=True, key="run_hotpotqa"
        )

    # Horizontal line to separate inputs from results
    st.markdown("---")

    # Create two columns for logs and results side by side with equal width
    log_col, results_col = st.columns(2)

    # Logs column
    with log_col:
        st.subheader("Evaluation Logs")
        logs_area = st.empty()

    # Results column
    with results_col:
        st.subheader("Evaluation Results")
        # Container for metrics
        results_metrics = st.container()
        # Container for Q&A pairs with scrollable area
        qa_container = st.container()
        with qa_container:
            st.write("Question-Answer Pairs")
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


def run_fever_evaluation():
    st.header("FEVER Evaluation")

    # Input section for configuration
    st.subheader("Configuration")

    # Create three columns for inputs
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # Input section for dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            value="datasets\\fever_train.jsonl",
            help="Path to the FEVER dataset JSONL file",
            key="fever_dataset_path",
        )

    with col2:
        # Input for number of questions/claims
        num_claims = st.number_input(
            "Number of Claims",
            min_value=0,
            value=1,
            help="Choose how many claims to sample from the dataset (0 for all)",
            key="fever_num_claims",
        )

    with col3:
        # Run evaluation button
        st.write("")  # Add some space
        run_button = st.button(
            "Run Evaluation", use_container_width=True, key="run_fever"
        )

    # Horizontal line to separate inputs from results
    st.markdown("---")

    # Create two columns for logs and results side by side with equal width
    log_col, results_col = st.columns(2)

    # Logs column
    with log_col:
        st.subheader("Evaluation Logs")
        logs_area = st.empty()

    # Results column
    with results_col:
        st.subheader("Evaluation Results")
        # Container for metrics
        results_metrics = st.container()
        # Container for claim-evidence pairs with scrollable area
        qa_container = st.container()
        with qa_container:
            st.write("Claim-Verification Pairs")
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
                fever_eval = FeverEval(dataset_path)
                fever_eval.load_fever_dataset()
                claims_to_evaluate = fever_eval.get_claims(num_claims)
                result = fever_eval.eval_claims(claims_to_evaluate)

            # Restore stdout
            sys.stdout = orig_stdout

            # Display results
            with results_metrics:
                st.metric(
                    "Correct Verifications",
                    result["correct_verifications"],
                    f"{result['correct_verifications']}/{len(claims_to_evaluate)} claims",
                )

            # Add CSS to create a scrollable container with fixed height (reusing the same style)
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

            # Generate markdown for claim verification pairs
            qa_text = ""
            for idx, claim_pair in enumerate(result["claim_verification_pairs"]):
                status = "✅ CORRECT" if claim_pair["correct"] else "❌ INCORRECT"
                qa_text += f"Claim {idx+1}: {claim_pair['claim']}\n\n"
                qa_text += f"Verification: {claim_pair['verification']}\n\n"
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
    run_streamlit_app()
