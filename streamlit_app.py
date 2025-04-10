import streamlit as st
import sys
from hotpotqa_eval import HotpotQAEval, StreamlitPrintCapture
from fever_eval import FeverEval
from alfworld_eval import ALFWorldEval


def run_streamlit_app():
    # Set page to wide mode to use the full screen width
    st.set_page_config(layout="wide", page_title="Evaluation Dashboard")

    st.title("Evaluation Dashboard")

    # Create tabs for HotpotQA, FEVER, and ALFWorld
    tab1, tab2, tab3 = st.tabs(
        ["HotpotQA Evaluation", "FEVER Evaluation", "ALFWorld Evaluation"]
    )

    with tab1:
        run_hotpotqa_evaluation()

    with tab2:
        run_fever_evaluation()

    with tab3:
        run_alfworld_evaluation()


def run_hotpotqa_evaluation():
    # Update CSS for new card design
    st.markdown(
        """
        <style>
        .eval-container {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 8px;
            margin-bottom: 5px;
        }
        .status-bubble {
            flex: 1;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            padding: 5px 10px;
            background: rgba(0,0,0,0.05);
            min-width: 180px;
        }
        .status-text {
            color: #4CAF50;
            font-size: 0.95em;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metrics-grid {
            display: flex;
            gap: 10px;
            margin: 0;
        }
        .metric-card {
            background: rgba(0,0,0,0.02);
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 8px 12px;
            width: 90px;
            text-align: center;
        }
        .metric-title {
            font-size: 0.75em;
            color: #666;
            margin-bottom: 4px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        .metric-correct {
            color: #4CAF50;
        }
        .metric-incorrect {
            color: #f44336;
        }
        .metric-percent {
            color: #2196F3;
        }
        .agent-label {
            font-size: 1em;
            font-weight: bold;
            margin-bottom: 5px;
            padding: 4px;
            border-radius: 4px;
            text-align: center;
        }
        .react-agent {
            background-color: rgba(33, 150, 243, 0.1);
            border: 1px solid #2196F3;
            color: #2196F3;
        }
        .gpt4o-agent {
            background-color: rgba(76, 175, 80, 0.1);
            border: 1px solid #4CAF50;
            color: #4CAF50;
        }
        .o3mini-agent {
            background-color: rgba(156, 39, 176, 0.1);
            border: 1px solid #9C27B0;
            color: #9C27B0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("HotpotQA Evaluation")

    # Create a compact configuration section
    with st.expander("Configuration", expanded=True):
        # Create columns for inputs
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])

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

        with col4:
            # Agent selection options
            st.write("Select agents to evaluate:")
            use_react = st.checkbox("React Agent", value=True, key="hotpotqa_use_react")
            use_gpt4o = st.checkbox(
                "GPT-4o Agent", value=True, key="hotpotqa_use_gpt4o"
            )
            use_o3mini = st.checkbox(
                "o3-mini Agent", value=True, key="hotpotqa_use_o3mini"
            )

    # Create a compact evaluation status bubble
    evaluation_expander = st.expander("Evaluation Status", expanded=True)
    with evaluation_expander:
        # Container for status and metrics
        status_container = st.empty()

    # Horizontal line to separate status from results
    st.markdown("---")

    # Results tabs - separate tabs for each selected agent
    results_tabs = []
    tab_titles = []

    if use_react:
        tab_titles.append("React Agent Results")
    if use_gpt4o:
        tab_titles.append("GPT-4o Results")
    if use_o3mini:
        tab_titles.append("o3-mini Results")

    # If no agents are selected, show a message
    if not tab_titles:
        st.warning("Please select at least one agent to evaluate.")
        return

    results_tabs = st.tabs(tab_titles)

    # Logs area - below the tabs
    logs_container = st.container()
    with logs_container:
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
                            <p style="color: #2196F3; font-size: 0.8em; margin: 2px 0 0 0;">Progress: {current_item}/{total_items}</p>
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
                        current_item=0,
                        total_items=len(questions_to_evaluate),
                        correct=0,
                        incorrect=0,
                        accuracy=0,
                    ),
                    unsafe_allow_html=True,
                )

                # Pass the agent selection flags to the eval_questions method
                original_eval_questions = hotpot_eval.eval_questions

                def eval_questions_with_progress(questions):
                    # Keep the original function's logic but update progress
                    for i, question in enumerate(questions):
                        # Update status with current agent and question
                        current_agent = (
                            f"Evaluating with {hotpot_eval.evaluation_progress['current_agent']}"
                            if hasattr(hotpot_eval, "evaluation_progress")
                            else "Evaluating"
                        )
                        status_container.markdown(
                            status_template.format(
                                status_text=f"{current_agent}: {question[:80]}...",
                                current_item=i + 1,
                                total_items=len(questions),
                                correct=0,
                                incorrect=0,
                                accuracy=0,
                            ),
                            unsafe_allow_html=True,
                        )

                    # Call original function
                    return original_eval_questions(questions)

                hotpot_eval.eval_questions = eval_questions_with_progress

                # Run evaluation with selected agents
                result = hotpot_eval.eval_questions(
                    questions_to_evaluate,
                    use_react=use_react,
                    use_gpt4o=use_gpt4o,
                    use_o3mini=use_o3mini,
                )

                # Prepare status displays for each selected agent
                status_displays = []

                # Calculate metrics for React agent if selected
                if use_react:
                    react_correct = result["react_results"]["correct_answers"]
                    total_questions = len(questions_to_evaluate)
                    react_incorrect = total_questions - react_correct
                    react_accuracy = (
                        int((react_correct / total_questions) * 100)
                        if total_questions > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label react-agent">React Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_questions,
                        total_items=total_questions,
                        correct=react_correct,
                        incorrect=react_incorrect,
                        accuracy=react_accuracy,
                    )}
                    """
                    )

                # Calculate metrics for GPT-4o agent if selected
                if use_gpt4o:
                    direct_correct = result["direct_results"]["correct_answers"]
                    total_questions = len(questions_to_evaluate)
                    direct_incorrect = total_questions - direct_correct
                    direct_accuracy = (
                        int((direct_correct / total_questions) * 100)
                        if total_questions > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label gpt4o-agent">GPT-4o Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_questions,
                        total_items=total_questions,
                        correct=direct_correct,
                        incorrect=direct_incorrect,
                        accuracy=direct_accuracy,
                    )}
                    """
                    )

                # Calculate metrics for o3-mini agent if selected
                if use_o3mini:
                    o3mini_correct = result["o3mini_results"]["correct_answers"]
                    total_questions = len(questions_to_evaluate)
                    o3mini_incorrect = total_questions - o3mini_correct
                    o3mini_accuracy = (
                        int((o3mini_correct / total_questions) * 100)
                        if total_questions > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label o3mini-agent">o3-mini Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_questions,
                        total_items=total_questions,
                        correct=o3mini_correct,
                        incorrect=o3mini_incorrect,
                        accuracy=o3mini_accuracy,
                    )}
                    """
                    )

                # Combined status display for all selected agents
                status_container.markdown(
                    "".join(status_displays),
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
                    background-color: rgba(0,0,0,0.02);
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            # Display results for each selected agent in its respective tab
            tab_index = 0

            # Generate markdown for React agent QA pairs if selected
            if use_react:
                with results_tabs[tab_index]:
                    st.subheader("React Agent Question-Answer Pairs")
                    react_qa_text = ""
                    for idx, qa_pair in enumerate(
                        result["react_results"]["question_answer_pairs"]
                    ):
                        status = "✅ VALID" if qa_pair["valid"] else "❌ INVALID"
                        react_qa_text += f"Question {idx+1}: {qa_pair['question']}\n\n"
                        react_qa_text += f"Answer: {qa_pair['answer']}\n\n"
                        react_qa_text += f"Status: {status}\n\n"
                        react_qa_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{react_qa_text}</div>',
                        unsafe_allow_html=True,
                    )
                tab_index += 1

            # Generate markdown for Direct GPT-4o agent QA pairs if selected
            if use_gpt4o:
                with results_tabs[tab_index]:
                    st.subheader("GPT-4o Question-Answer Pairs")
                    direct_qa_text = ""
                    for idx, qa_pair in enumerate(
                        result["direct_results"]["question_answer_pairs"]
                    ):
                        status = "✅ VALID" if qa_pair["valid"] else "❌ INVALID"
                        direct_qa_text += f"Question {idx+1}: {qa_pair['question']}\n\n"
                        direct_qa_text += f"Answer: {qa_pair['answer']}\n\n"
                        direct_qa_text += f"Status: {status}\n\n"
                        direct_qa_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{direct_qa_text}</div>',
                        unsafe_allow_html=True,
                    )
                tab_index += 1

            # Generate markdown for o3-mini agent QA pairs if selected
            if use_o3mini:
                with results_tabs[tab_index]:
                    st.subheader("o3-mini Question-Answer Pairs")
                    o3mini_qa_text = ""
                    for idx, qa_pair in enumerate(
                        result["o3mini_results"]["question_answer_pairs"]
                    ):
                        status = "✅ VALID" if qa_pair["valid"] else "❌ INVALID"
                        o3mini_qa_text += f"Question {idx+1}: {qa_pair['question']}\n\n"
                        o3mini_qa_text += f"Answer: {qa_pair['answer']}\n\n"
                        o3mini_qa_text += f"Status: {status}\n\n"
                        o3mini_qa_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{o3mini_qa_text}</div>',
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


def run_fever_evaluation():
    # Reuse the same CSS styling from HotpotQA function for consistency

    st.header("FEVER Evaluation")

    # Create a compact configuration section
    with st.expander("Configuration", expanded=True):
        # Create four columns for inputs
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])

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

        with col4:
            # Agent selection options
            st.write("Select agents to evaluate:")
            use_react = st.checkbox("React Agent", value=True, key="fever_use_react")
            use_gpt4o = st.checkbox("GPT-4o Agent", value=True, key="fever_use_gpt4o")
            use_o3mini = st.checkbox(
                "o3-mini Agent", value=True, key="fever_use_o3mini"
            )

    # Create a compact evaluation status bubble
    evaluation_expander = st.expander("Evaluation Status", expanded=True)
    with evaluation_expander:
        # Container for status and metrics
        status_container = st.empty()

    # Horizontal line to separate status from results
    st.markdown("---")

    # Results tabs - separate tabs for each selected agent
    results_tabs = []
    tab_titles = []

    if use_react:
        tab_titles.append("React Agent Results")
    if use_gpt4o:
        tab_titles.append("GPT-4o Results")
    if use_o3mini:
        tab_titles.append("o3-mini Results")

    # If no agents are selected, show a message
    if not tab_titles:
        st.warning("Please select at least one agent to evaluate.")
        return

    results_tabs = st.tabs(tab_titles)

    # Logs area - below the tabs
    logs_container = st.container()
    with logs_container:
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
                            <p style="color: #2196F3; font-size: 0.8em; margin: 2px 0 0 0;">Progress: {current_item}/{total_items}</p>
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
                        current_item=0,
                        total_items=len(claims_to_evaluate),
                        correct=0,
                        incorrect=0,
                        accuracy=0,
                    ),
                    unsafe_allow_html=True,
                )

                # Capture the original method to monkey patch it
                original_eval_claims = fever_eval.eval_claims

                def eval_claims_with_progress(claim_label_pairs):
                    # Let the original function handle everything
                    for i, (claim, _) in enumerate(claim_label_pairs):
                        # Update status with current agent and claim
                        current_agent = (
                            fever_eval.evaluation_progress["current_agent"]
                            if hasattr(fever_eval, "evaluation_progress")
                            else "Evaluating"
                        )
                        status_container.markdown(
                            status_template.format(
                                status_text=f"{current_agent} claim {i+1}: {claim[:80]}...",
                                current_item=i + 1,
                                total_items=len(claim_label_pairs),
                                correct=0,
                                incorrect=0,
                                accuracy=0,
                            ),
                            unsafe_allow_html=True,
                        )

                    result = original_eval_claims(
                        claim_label_pairs, use_react, use_gpt4o, use_o3mini
                    )
                    return result

                fever_eval.eval_claims = eval_claims_with_progress

                # Run the evaluation with selected agents
                result = fever_eval.eval_claims(
                    claims_to_evaluate, use_react, use_gpt4o, use_o3mini
                )

                # Prepare status displays for each selected agent
                status_displays = []

                # Calculate metrics for React agent if selected
                if use_react:
                    react_correct = result["react_results"]["correct_verifications"]
                    total_claims = len(claims_to_evaluate)
                    react_incorrect = total_claims - react_correct
                    react_accuracy = (
                        int((react_correct / total_claims) * 100)
                        if total_claims > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label react-agent">React Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_claims,
                        total_items=total_claims,
                        correct=react_correct,
                        incorrect=react_incorrect,
                        accuracy=react_accuracy,
                    )}
                    """
                    )

                # Calculate metrics for GPT-4o agent if selected
                if use_gpt4o:
                    direct_correct = result["direct_results"]["correct_verifications"]
                    total_claims = len(claims_to_evaluate)
                    direct_incorrect = total_claims - direct_correct
                    direct_accuracy = (
                        int((direct_correct / total_claims) * 100)
                        if total_claims > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label gpt4o-agent">GPT-4o Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_claims,
                        total_items=total_claims,
                        correct=direct_correct,
                        incorrect=direct_incorrect,
                        accuracy=direct_accuracy,
                    )}
                    """
                    )

                # Calculate metrics for o3-mini agent if selected
                if use_o3mini:
                    o3mini_correct = result["o3mini_results"]["correct_verifications"]
                    total_claims = len(claims_to_evaluate)
                    o3mini_incorrect = total_claims - o3mini_correct
                    o3mini_accuracy = (
                        int((o3mini_correct / total_claims) * 100)
                        if total_claims > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label o3mini-agent">o3-mini Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_claims,
                        total_items=total_claims,
                        correct=o3mini_correct,
                        incorrect=o3mini_incorrect,
                        accuracy=o3mini_accuracy,
                    )}
                    """
                    )

                # Combined status display for all selected agents
                status_container.markdown(
                    "".join(status_displays),
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
                    background-color: rgba(0,0,0,0.02);
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            # Display results for each selected agent in its respective tab
            tab_index = 0

            # Generate markdown for React agent claim verification pairs if selected
            if use_react:
                with results_tabs[tab_index]:
                    st.subheader("React Agent Claim-Verification Pairs")
                    react_qa_text = ""
                    for idx, claim_pair in enumerate(
                        result["react_results"]["claim_verification_pairs"]
                    ):
                        status = (
                            "✅ CORRECT" if claim_pair["correct"] else "❌ INCORRECT"
                        )
                        react_qa_text += f"Claim {idx+1}: {claim_pair['claim']}\n\n"
                        react_qa_text += (
                            f"Ground Truth: {claim_pair['ground_truth']}\n\n"
                        )
                        react_qa_text += (
                            f"Verification: {claim_pair['verification']}\n\n"
                        )
                        react_qa_text += (
                            f"Evidence: {claim_pair.get('evidence', '')}\n\n"
                        )
                        react_qa_text += f"Status: {status}\n\n"
                        react_qa_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{react_qa_text}</div>',
                        unsafe_allow_html=True,
                    )
                tab_index += 1

            # Generate markdown for GPT-4o agent claim verification pairs if selected
            if use_gpt4o:
                with results_tabs[tab_index]:
                    st.subheader("GPT-4o Claim-Verification Pairs")
                    direct_qa_text = ""
                    for idx, claim_pair in enumerate(
                        result["direct_results"]["claim_verification_pairs"]
                    ):
                        status = (
                            "✅ CORRECT" if claim_pair["correct"] else "❌ INCORRECT"
                        )
                        direct_qa_text += f"Claim {idx+1}: {claim_pair['claim']}\n\n"
                        direct_qa_text += (
                            f"Ground Truth: {claim_pair['ground_truth']}\n\n"
                        )
                        direct_qa_text += (
                            f"Verification: {claim_pair['verification']}\n\n"
                        )
                        direct_qa_text += (
                            f"Evidence: {claim_pair.get('evidence', '')}\n\n"
                        )
                        direct_qa_text += f"Status: {status}\n\n"
                        direct_qa_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{direct_qa_text}</div>',
                        unsafe_allow_html=True,
                    )
                tab_index += 1

            # Generate markdown for o3-mini agent claim verification pairs if selected
            if use_o3mini:
                with results_tabs[tab_index]:
                    st.subheader("o3-mini Claim-Verification Pairs")
                    o3mini_qa_text = ""
                    for idx, claim_pair in enumerate(
                        result["o3mini_results"]["claim_verification_pairs"]
                    ):
                        status = (
                            "✅ CORRECT" if claim_pair["correct"] else "❌ INCORRECT"
                        )
                        o3mini_qa_text += f"Claim {idx+1}: {claim_pair['claim']}\n\n"
                        o3mini_qa_text += (
                            f"Ground Truth: {claim_pair['ground_truth']}\n\n"
                        )
                        o3mini_qa_text += (
                            f"Verification: {claim_pair['verification']}\n\n"
                        )
                        o3mini_qa_text += (
                            f"Evidence: {claim_pair.get('evidence', '')}\n\n"
                        )
                        o3mini_qa_text += f"Status: {status}\n\n"
                        o3mini_qa_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{o3mini_qa_text}</div>',
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


def run_alfworld_evaluation():
    """
    Run ALFWorld evaluation in the Streamlit app.
    """
    st.header("ALFWorld Evaluation")

    # Create a compact configuration section
    with st.expander("Configuration", expanded=True):
        # Create columns for inputs
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])

        with col1:
            # Input section for dataset path
            dataset_path = st.text_input(
                "Dataset Path",
                value="datasets\\alfworld",
                help="Path to the ALFWorld dataset directory",
                key="alfworld_dataset_path",
            )

        with col2:
            # Input for number of tasks
            num_tasks = st.number_input(
                "Number of Tasks",
                min_value=0,
                value=1,
                help="Choose how many tasks to sample from the dataset (0 for all)",
                key="alfworld_num_tasks",
            )

        with col3:
            # Run evaluation button
            st.write("")  # Add some space
            run_button = st.button(
                "Run Evaluation", use_container_width=True, key="run_alfworld"
            )

        with col4:
            # Agent selection options
            st.write("Select agents to evaluate:")
            use_react = st.checkbox("React Agent", value=True, key="alfworld_use_react")
            use_gpt4o = st.checkbox(
                "GPT-4o Agent", value=True, key="alfworld_use_gpt4o"
            )
            use_o3mini = st.checkbox(
                "o3-mini Agent", value=True, key="alfworld_use_o3mini"
            )

    # Create evaluation status section
    evaluation_expander = st.expander("Evaluation Status", expanded=True)
    with evaluation_expander:
        # Container for status and metrics
        status_container = st.empty()

    # Horizontal line to separate status from results
    st.markdown("---")

    # Results tabs - separate tabs for each selected agent
    results_tabs = []
    tab_titles = []

    if use_react:
        tab_titles.append("React Agent Results")
    if use_gpt4o:
        tab_titles.append("GPT-4o Results")
    if use_o3mini:
        tab_titles.append("o3-mini Results")

    # If no agents are selected, show a message
    if not tab_titles:
        st.warning("Please select at least one agent to evaluate.")
        return

    results_tabs = st.tabs(tab_titles)

    # Logs area - below the tabs
    logs_container = st.container()
    with logs_container:
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
                alfworld_eval = ALFWorldEval(dataset_path)
                alfworld_eval.load_alfworld_dataset()
                tasks_to_evaluate = alfworld_eval.get_tasks(num_tasks)

                # Display initial progress
                status_template = """
                    <div class="eval-container">
                        <div class="status-bubble">
                            <p class="status-text">{status_text}</p>
                            <p style="color: #2196F3; font-size: 0.8em; margin: 2px 0 0 0;">Progress: {current_item}/{total_items}</p>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-title">SUCCESSFUL</div>
                                <div class="metric-value metric-correct">{successful}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">FAILED</div>
                                <div class="metric-value metric-incorrect">{failed}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">SUCCESS %</div>
                                <div class="metric-value metric-percent">{success_rate}%</div>
                            </div>
                        </div>
                    </div>
                """

                status_container.markdown(
                    status_template.format(
                        status_text="Initializing evaluation...",
                        current_item=0,
                        total_items=len(tasks_to_evaluate),
                        successful=0,
                        failed=0,
                        success_rate=0,
                    ),
                    unsafe_allow_html=True,
                )

                # Capture the original method to monkey patch it
                original_eval_tasks = alfworld_eval.eval_tasks

                def eval_tasks_with_progress(
                    tasks, use_react=True, use_gpt4o=True, use_o3mini=True
                ):
                    # Return the evaluation results but update progress during execution
                    for i, task in enumerate(tasks):
                        task_description = alfworld_eval.extract_task_description(task)
                        task_type = task["task_type"]

                        # Update status with current task
                        current_agent = "Evaluating"
                        if hasattr(alfworld_eval, "evaluation_progress"):
                            if "current_agent" in alfworld_eval.evaluation_progress:
                                current_agent = alfworld_eval.evaluation_progress[
                                    "current_agent"
                                ]

                        status_container.markdown(
                            status_template.format(
                                status_text=f"{current_agent} - {task_type}: {task_description[:80]}...",
                                current_item=i + 1,
                                total_items=len(tasks),
                                successful=0,
                                failed=0,
                                success_rate=0,
                            ),
                            unsafe_allow_html=True,
                        )

                    result = original_eval_tasks(
                        tasks, use_react, use_gpt4o, use_o3mini
                    )
                    return result

                # Replace the original method with our instrumented version
                alfworld_eval.eval_tasks = eval_tasks_with_progress

                # Run the evaluation with selected agents
                result = alfworld_eval.eval_tasks(
                    tasks_to_evaluate, use_react, use_gpt4o, use_o3mini
                )

                # Prepare status displays for each selected agent
                status_displays = []

                # Calculate metrics for React agent if selected
                if use_react:
                    react_successful = result["react_results"]["successful_tasks"]
                    total_tasks = len(tasks_to_evaluate)
                    react_failed = total_tasks - react_successful
                    react_success_rate = (
                        int((react_successful / total_tasks) * 100)
                        if total_tasks > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label react-agent">React Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_tasks,
                        total_items=total_tasks,
                        successful=react_successful,
                        failed=react_failed,
                        success_rate=react_success_rate,
                    )}
                    """
                    )

                # Calculate metrics for GPT-4o agent if selected
                if use_gpt4o:
                    direct_successful = result["direct_results"]["successful_tasks"]
                    total_tasks = len(tasks_to_evaluate)
                    direct_failed = total_tasks - direct_successful
                    direct_success_rate = (
                        int((direct_successful / total_tasks) * 100)
                        if total_tasks > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label gpt4o-agent">GPT-4o Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_tasks,
                        total_items=total_tasks,
                        successful=direct_successful,
                        failed=direct_failed,
                        success_rate=direct_success_rate,
                    )}
                    """
                    )

                # Calculate metrics for o3-mini agent if selected
                if use_o3mini:
                    o3mini_successful = result["o3mini_results"]["successful_tasks"]
                    total_tasks = len(tasks_to_evaluate)
                    o3mini_failed = total_tasks - o3mini_successful
                    o3mini_success_rate = (
                        int((o3mini_successful / total_tasks) * 100)
                        if total_tasks > 0
                        else 0
                    )

                    status_displays.append(
                        f"""
                    <div class="agent-label o3mini-agent">o3-mini Agent Results</div>
                    {status_template.format(
                        status_text="Evaluation completed",
                        current_item=total_tasks,
                        total_items=total_tasks,
                        successful=o3mini_successful,
                        failed=o3mini_failed,
                        success_rate=o3mini_success_rate,
                    )}
                    """
                    )

                # Combined status display for all selected agents
                status_container.markdown(
                    "".join(status_displays),
                    unsafe_allow_html=True,
                )

            # Restore stdout
            sys.stdout = orig_stdout

            # Add CSS for the scrollable container and task type tag
            st.markdown(
                """
                <style>
                .scrollable-container {
                    height: 400px;
                    overflow-y: auto;
                    border: 1px solid #e6e6e6;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: rgba(0,0,0,0.02);
                }
                .task-type-tag {
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    font-weight: bold;
                    margin-right: 8px;
                    background-color: #2c3e50;
                    color: white;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            # Display results for each selected agent in its respective tab
            tab_index = 0

            # Generate markdown for React agent task results if selected
            if use_react:
                with results_tabs[tab_index]:
                    st.subheader("React Agent Task Results")
                    react_task_text = ""
                    for idx, task_result in enumerate(
                        result["react_results"]["task_results"]
                    ):
                        status = (
                            "✅ SUCCESSFUL" if task_result["success"] else "❌ FAILED"
                        )
                        task_type = task_result["task_type"]
                        react_task_text += f"<div class='task-type-tag'>{task_type}</div> <b>Task {idx+1}</b>: {task_result['task_description']}\n\n"
                        react_task_text += f"Actions: {', '.join(task_result.get('actions', ['No actions recorded']))}\n\n"
                        react_task_text += f"Reasoning: {task_result.get('reasoning', 'No reasoning provided')}\n\n"
                        react_task_text += f"Status: {status}\n\n"
                        react_task_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{react_task_text}</div>',
                        unsafe_allow_html=True,
                    )
                tab_index += 1

            # Generate markdown for GPT-4o agent task results if selected
            if use_gpt4o:
                with results_tabs[tab_index]:
                    st.subheader("GPT-4o Task Results")
                    direct_task_text = ""
                    for idx, task_result in enumerate(
                        result["direct_results"]["task_results"]
                    ):
                        status = (
                            "✅ SUCCESSFUL" if task_result["success"] else "❌ FAILED"
                        )
                        task_type = task_result["task_type"]
                        direct_task_text += f"<div class='task-type-tag'>{task_type}</div> <b>Task {idx+1}</b>: {task_result['task_description']}\n\n"
                        direct_task_text += f"Actions: {', '.join(task_result.get('actions', ['No actions recorded']))}\n\n"
                        direct_task_text += f"Reasoning: {task_result.get('reasoning', 'No reasoning provided')}\n\n"
                        direct_task_text += f"Status: {status}\n\n"
                        direct_task_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{direct_task_text}</div>',
                        unsafe_allow_html=True,
                    )
                tab_index += 1

            # Generate markdown for o3-mini agent task results if selected
            if use_o3mini:
                with results_tabs[tab_index]:
                    st.subheader("o3-mini Task Results")
                    o3mini_task_text = ""
                    for idx, task_result in enumerate(
                        result["o3mini_results"]["task_results"]
                    ):
                        status = (
                            "✅ SUCCESSFUL" if task_result["success"] else "❌ FAILED"
                        )
                        task_type = task_result["task_type"]
                        o3mini_task_text += f"<div class='task-type-tag'>{task_type}</div> <b>Task {idx+1}</b>: {task_result['task_description']}\n\n"
                        o3mini_task_text += f"Actions: {', '.join(task_result.get('actions', ['No actions recorded']))}\n\n"
                        o3mini_task_text += f"Reasoning: {task_result.get('reasoning', 'No reasoning provided')}\n\n"
                        o3mini_task_text += f"Status: {status}\n\n"
                        o3mini_task_text += "---\n\n"

                    # Display the markdown in a scrollable div
                    st.markdown(
                        f'<div class="scrollable-container">{o3mini_task_text}</div>',
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            sys.stdout = orig_stdout
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_streamlit_app()
