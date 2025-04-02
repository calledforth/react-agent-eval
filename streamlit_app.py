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

    # Create a status area for showing progress
    progress_container = st.container()
    with progress_container:
        progress_col1, progress_col2 = st.columns(2)
        with progress_col1:
            progress_status = st.empty()
        with progress_col2:
            progress_metrics = st.empty()

    # Results section - we'll put results above logs now
    results_container = st.container()
    with results_container:
        result_col1, result_col2 = st.columns(2)

        # Results column (left side)
        with result_col1:
            st.subheader("Evaluation Results")
            # Container for metrics
            results_metrics = st.container()
            # Container for Q&A pairs with scrollable area
            qa_container = st.container()
            with qa_container:
                st.write("Question-Answer Pairs")
                # Create a container with fixed height and scrolling
                qa_scroll_container = st.container()

        # Logs column (right side)
        with result_col2:
            st.subheader("Evaluation Logs")
            logs_area = st.empty()

    # Run evaluation when button is clicked
    if run_button:
        try:
            # Set up logging to capture output
            orig_stdout = sys.stdout
            log_capture = StreamlitPrintCapture(logs_area)
            sys.stdout = log_capture

            # Create progress tracker
            progress_placeholder = progress_status.empty()
            progress_metrics_placeholder = progress_metrics.empty()

            # Run the evaluation
            with st.spinner("Evaluation in progress..."):
                hotpot_eval = HotpotQAEval(dataset_path)
                hotpot_eval.load_hotpotqa_dataset()
                questions_to_evaluate = hotpot_eval.get_questions(num_questions)

                # Display initial progress
                progress_placeholder.info(
                    f"Starting evaluation of {len(questions_to_evaluate)} questions..."
                )

                # Monkey patch the eval_questions method to update progress
                original_eval_questions = hotpot_eval.eval_questions

                def eval_questions_with_progress(questions):
                    for i, question in enumerate(questions):
                        # Update progress display
                        progress_placeholder.info(
                            f"Evaluating question {i+1}/{len(questions)}: {question[:100]}..."
                        )
                        progress_metrics_placeholder.metric(
                            "Progress",
                            f"{i+1}/{len(questions)}",
                            f"{int((i+1)/len(questions)*100)}%",
                        )

                    return original_eval_questions(questions)

                hotpot_eval.eval_questions = eval_questions_with_progress

                # Run evaluation
                result = hotpot_eval.eval_questions(questions_to_evaluate)

                # Final progress update
                progress_placeholder.success(f"Evaluation completed successfully!")
                progress_metrics_placeholder.metric(
                    "Progress",
                    f"{len(questions_to_evaluate)}/{len(questions_to_evaluate)}",
                    "100%",
                )

            # Restore stdout
            sys.stdout = orig_stdout

            # Display results
            # with results_metrics:
            #     st.metric(
            #         "Correct Answers",
            #         result["correct_answers"],
            #         f"{result['correct_answers']}/{len(questions_to_evaluate)} questions",
            #     )

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

    # Create a status area for showing progress
    progress_container = st.container()
    with progress_container:
        progress_col1, progress_col2 = st.columns(2)
        with progress_col1:
            progress_status = st.empty()
        with progress_col2:
            progress_metrics = st.empty()

    # Results section - we'll put results above logs now
    results_container = st.container()
    with results_container:
        result_col1, result_col2 = st.columns(2)

        # Results column (left side)
        with result_col1:
            st.subheader("Evaluation Results")
            # Container for metrics
            results_metrics = st.container()
            # Container for claim-evidence pairs with scrollable area
            qa_container = st.container()
            with qa_container:
                st.write("Claim-Verification Pairs")
                # Create a container with fixed height and scrolling
                qa_scroll_container = st.container()

        # Logs column (right side)
        with result_col2:
            st.subheader("Evaluation Logs")
            logs_area = st.empty()

    # Run evaluation when button is clicked
    if run_button:
        try:
            # Set up logging to capture output
            orig_stdout = sys.stdout
            log_capture = StreamlitPrintCapture(logs_area)
            sys.stdout = log_capture

            # Create progress tracker
            progress_placeholder = progress_status.empty()
            progress_metrics_placeholder = progress_metrics.empty()

            # Run the evaluation
            with st.spinner("Evaluation in progress..."):
                fever_eval = FeverEval(dataset_path)
                fever_eval.load_fever_dataset()
                claims_to_evaluate = fever_eval.get_claims(num_claims)

                # Display initial progress
                progress_placeholder.info(
                    f"Starting evaluation of {len(claims_to_evaluate)} claims..."
                )

                # Create a placeholder for the current claim and thinking round
                evaluation_status = st.empty()

                # Keep track of current claim and thinking round
                current_claim = {"index": 0, "text": "", "thinking_round": 0}

                # Capture the original method to monkey patch it
                original_eval_claims = fever_eval.eval_claims

                def eval_claims_with_progress(claim_label_pairs):
                    result = original_eval_claims(claim_label_pairs)

                    # Update the progress display based on the evaluation_progress in the result
                    for i, (claim, _) in enumerate(claim_label_pairs):
                        # Update progress metrics
                        progress_placeholder.info(
                            f"Evaluating claim {i+1}/{len(claim_label_pairs)}"
                        )
                        progress_metrics_placeholder.metric(
                            "Progress",
                            f"{i+1}/{len(claim_label_pairs)}",
                            f"{int((i+1)/len(claim_label_pairs)*100)}%",
                        )

                        # Update current claim tracking
                        current_claim["index"] = i + 1
                        current_claim["text"] = claim

                        # Update the evaluation status with current claim info
                        evaluation_status.info(
                            f"Claim {i+1}/{len(claim_label_pairs)}: {claim[:100]}..."
                        )

                    # Final progress update
                    progress_placeholder.success(f"Evaluation completed successfully!")
                    progress_metrics_placeholder.metric(
                        "Progress / Evaluation Results",
                        f"{len(claim_label_pairs)}/{len(claim_label_pairs)}",
                        "100%",
                    )

                    return result

                # Replace the method with our progress-tracking version
                fever_eval.eval_claims = eval_claims_with_progress

                # Run the evaluation
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
                qa_text += f"Ground Truth: {claim_pair['ground_truth']}\n\n"
                qa_text += f"Verification: {claim_pair['verification']}\n\n"
                qa_text += f"Evidence: {claim_pair.get('evidence', '')}\n\n"
                qa_text += f"Status: {status}\n\n"
                qa_text += "---\n\n"

            # Display the markdown in a scrollable div
            with qa_scroll_container:
                st.markdown(
                    f'<div class="scrollable-container">{qa_text}</div>',
                    unsafe_allow_html=True,
                )

            # st.success("Evaluation completed!")

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_streamlit_app()
