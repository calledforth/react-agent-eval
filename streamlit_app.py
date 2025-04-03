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
    # Update CSS for new card design
    st.markdown(
        """
        <style>
        .eval-container {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 10px;
        }
        .status-bubble {
            flex: 1;
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 8px 15px;
            background: #000000;
            min-width: 200px;
        }
        .status-text {
            color: #4CAF50;
            font-size: 1.1em;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metrics-grid {
            display: flex;
            gap: 15px;
            margin: 0;
        }
        .metric-card {
            background: #000000;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px 20px;
            width: 120px;
            text-align: center;
        }
        .metric-title {
            font-size: 0.9em;
            color: #888;
            margin-bottom: 8px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 1.4em;
            font-weight: bold;
        }
        .metric-correct {
            color: #4CAF50;
        }
        .metric-incorrect {
            color: #f44336;
        }
        .metric-percent {
            color: #4CAF50;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("HotpotQA Evaluation")

    # Create a compact configuration section
    with st.expander("Configuration", expanded=True):
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

    # Create a compact evaluation status bubble
    evaluation_expander = st.expander("Evaluation Status", expanded=True)
    with evaluation_expander:
        # Container for status and metrics
        status_container = st.empty()

    # Horizontal line to separate status from results
    st.markdown("---")

    # Results section - we'll put results above logs now
    results_container = st.container()
    with results_container:
        result_col1, result_col2 = st.columns(2)

        # Results column (left side)
        with result_col1:
            st.subheader("Evaluation Results")
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

            # Run the evaluation
            with st.spinner("Evaluation in progress..."):
                hotpot_eval = HotpotQAEval(dataset_path)
                hotpot_eval.load_hotpotqa_dataset()
                questions_to_evaluate = hotpot_eval.get_questions(num_questions)

                # Display initial progress
                status_template = """
                    <div class="eval-container">
                        <div class="status-bubble">
                            <p class="status-text">{status_text}</p>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-title">CORRECT</div>
                                <div class="metric-value metric-correct">{correct}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">INCORRECT</div>
                                <div class="metric-value metric-incorrect">{incorrect}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">ACCURACY</div>
                                <div class="metric-value metric-percent">{accuracy}%</div>
                            </div>
                        </div>
                    </div>
                """

                status_container.markdown(
                    status_template.format(
                        status_text="Initializing evaluation...",
                        correct=0,
                        incorrect=0,
                        accuracy=0,
                    ),
                    unsafe_allow_html=True,
                )

                # Monkey patch the eval_questions method to update progress
                original_eval_questions = hotpot_eval.eval_questions

                def eval_questions_with_progress(questions):
                    result = []
                    correct_count = 0

                    for i, question in enumerate(questions):
                        # Update status and metrics with new design
                        status_container.markdown(
                            status_template.format(
                                status_text="Evaluating: " + question[:100] + "...",
                                correct=correct_count,
                                incorrect=i + 1 - correct_count,
                                accuracy=(
                                    int(correct_count / (i + 1) * 100) if i > 0 else 0
                                ),
                            ),
                            unsafe_allow_html=True,
                        )

                    return original_eval_questions(questions)

                hotpot_eval.eval_questions = eval_questions_with_progress

                # Run evaluation
                result = hotpot_eval.eval_questions(questions_to_evaluate)

                # Final accuracy update
                correct_answers = result.get("correct_answers", 0)
                incorrect_answers = len(questions_to_evaluate) - correct_answers
                accuracy_percentage = (
                    int((correct_answers / len(questions_to_evaluate)) * 100)
                    if questions_to_evaluate
                    else 0
                )

                status_container.markdown(
                    status_template.format(
                        status_text="Evaluation completed!",
                        correct=correct_answers,
                        incorrect=incorrect_answers,
                        accuracy=accuracy_percentage,
                    ),
                    unsafe_allow_html=True,
                )

            # Restore stdout
            sys.stdout = orig_stdout

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

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


def run_fever_evaluation():
    # Update CSS for new card design
    st.markdown(
        """
        <style>
        .eval-container {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 10px;
        }
        .status-bubble {
            flex: 1;
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 8px 8px;
            background: transparent;
            min-width: 200px;
        }
        .status-text {
            color: #4CAF50;
            font-size: 1.1em;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metrics-grid {
            display: flex;
            gap: 15px;
            margin: 0;
        }
        .metric-card {
            background: transparent;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px 10px;
            width: 120px;
            text-align: center;
        }
        .metric-title {
            font-size: 0.9em;
            color: #888;
            margin-bottom: 8px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 1.4em;
            font-weight: bold;
        }
        .metric-correct {
            color: #4CAF50;
        }
        .metric-incorrect {
            color: #f44336;
        }
        .metric-percent {
            color: #4CAF50;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("FEVER Evaluation")

    # Create a compact configuration section
    with st.expander("Configuration", expanded=True):
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

    # Create a compact evaluation status bubble
    evaluation_expander = st.expander("Evaluation Status", expanded=True)
    with evaluation_expander:
        # Container for status and metrics
        status_container = st.empty()

    # Horizontal line to separate status from results
    st.markdown("---")

    # Results section - we'll put results above logs now
    results_container = st.container()
    with results_container:
        result_col1, result_col2 = st.columns(2)

        # Results column (left side)
        with result_col1:
            st.subheader("Evaluation Results")
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

            # Run the evaluation
            with st.spinner("Evaluation in progress..."):
                fever_eval = FeverEval(dataset_path)
                fever_eval.load_fever_dataset()
                claims_to_evaluate = fever_eval.get_claims(num_claims)

                # Display initial progress
                status_template = """
                    <div class="eval-container">
                        <div class="status-bubble">
                            <p class="status-text">{status_text}</p>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-title">CORRECT</div>
                                <div class="metric-value metric-correct">{correct}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">INCORRECT</div>
                                <div class="metric-value metric-incorrect">{incorrect}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">ACCURACY</div>
                                <div class="metric-value metric-percent">{accuracy}%</div>
                            </div>
                        </div>
                    </div>
                """

                status_container.markdown(
                    status_template.format(
                        status_text="Initializing evaluation...",
                        correct=0,
                        incorrect=0,
                        accuracy=0,
                    ),
                    unsafe_allow_html=True,
                )

                # Capture the original method to monkey patch it
                original_eval_claims = fever_eval.eval_claims

                def eval_claims_with_progress(claim_label_pairs):
                    correct_count = 0

                    for i, (claim, _) in enumerate(claim_label_pairs):
                        # Update status and metrics with new design
                        status_container.markdown(
                            status_template.format(
                                status_text="Evaluating claim {i} "
                                + claim[:100]
                                + "...",
                                correct=correct_count,
                                incorrect=i + 1 - correct_count,
                                accuracy=(
                                    int(correct_count / (i + 1) * 100) if i > 0 else 0
                                ),
                            ),
                            unsafe_allow_html=True,
                        )

                    result = original_eval_claims(claim_label_pairs)

                    # Update final accuracy
                    correct_verifications = result.get("correct_verifications", 0)
                    incorrect_verifications = (
                        len(claim_label_pairs) - correct_verifications
                    )
                    accuracy_percentage = (
                        int((correct_verifications / len(claim_label_pairs)) * 100)
                        if claim_label_pairs
                        else 0
                    )

                    status_container.markdown(
                        status_template.format(
                            status_text="Evaluation completed!",
                            correct=correct_verifications,
                            incorrect=incorrect_verifications,
                            accuracy=accuracy_percentage,
                        ),
                        unsafe_allow_html=True,
                    )

                    return result

                fever_eval.eval_claims = eval_claims_with_progress

                # Run the evaluation
                result = fever_eval.eval_claims(claims_to_evaluate)

            # Restore stdout
            sys.stdout = orig_stdout

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

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_streamlit_app()
