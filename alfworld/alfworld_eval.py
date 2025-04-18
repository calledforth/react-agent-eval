import os
import random
import json
from typing import List, Dict, Any, Tuple
import re
from Agent import Agent


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


class ALFWorldEval:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.task_types = []
        self.tasks = []
        self.length = 0
        self.direct_success_count = 0
        self.react_success_count = 0
        self.o3mini_success_count = 0
        self.direct_task_results = []
        self.react_task_results = []
        self.o3mini_task_results = []
        # Progress tracking for Streamlit
        self.evaluation_progress = {
            "current_task": 0,
            "total_tasks": 0,
            "current_agent": "",
            "thinking_round": 0,
            "status": "",
            "task_type": "",
            "task_description": "",
        }

    def load_alfworld_dataset(self):
        """
        Load the ALFWorld dataset structure from the dataset path.

        Returns:
            String message indicating success
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        # Get task types by looking at subdirectories of train directory
        json_dir = os.path.join(self.dataset_path, "json_2.1.1")
        train_dir = os.path.join(json_dir, "train")

        # Extract task types from directory names
        task_dirs = [d for d in os.listdir(train_dir)]
        self.task_types = list(
            set(
                [
                    d.split("-")[0]
                    for d in task_dirs
                    if os.path.isdir(os.path.join(train_dir, d))
                ]
            )
        )

        # For each task type, get some tasks
        for task_type in self.task_types:
            matching_tasks = [d for d in task_dirs if d.startswith(task_type)]
            # Add tasks with their full paths
            for task in matching_tasks:
                task_path = os.path.join(train_dir, task)
                if os.path.isdir(task_path):
                    self.tasks.append(
                        {"task_type": task_type, "task_name": task, "path": task_path}
                    )

        self.length = len(self.tasks)

        return f"Successfully loaded ALFWorld dataset with {self.length} tasks of {len(self.task_types)} types"

    def get_tasks(self, num_tasks: int = 5) -> List[Dict[str, str]]:
        """
        Get a specified number of tasks from the dataset.
        If num_tasks is 0, returns all tasks.
        If num_tasks > 0, returns that many randomly selected tasks.

        Args:
            num_tasks (int): Number of tasks to retrieve. 0 for all tasks.

        Returns:
            List of task dictionaries
        """
        if num_tasks == 0:
            return self.tasks

        elif num_tasks > self.length:
            raise ValueError(
                f"Requested number of tasks exceeds dataset size ({self.length})"
            )

        return random.sample(self.tasks, num_tasks)

    def extract_task_description(self, task: Dict[str, str]) -> str:
        """
        Extract a human-readable description from the task.

        Args:
            task (Dict): Task dictionary with path and other metadata

        Returns:
            String describing the task
        """
        task_name = task["task_name"]
        task_parts = task_name.split("-")
        task_type = task_parts[0]

        if task_type == "look_at_obj_in_light":
            object_name = task_parts[1]
            light_source = task_parts[3]
            return f"Look at {object_name} in the light of a {light_source}"

        elif task_type == "pick_and_place_simple":
            object_name = task_parts[1]
            target_name = task_parts[3]
            return f"Pick up the {object_name} and place it on/in the {target_name}"

        elif task_type == "pick_clean_then_place_in_recep":
            object_name = task_parts[1]
            target_name = task_parts[3]
            return f"Pick up the {object_name}, clean it, and place it in/on the {target_name}"

        elif task_type == "pick_cool_then_place_in_recep":
            object_name = task_parts[1]
            target_name = task_parts[3]
            return f"Pick up the {object_name}, cool it, and place it in/on the {target_name}"

        elif task_type == "pick_heat_then_place_in_recep":
            object_name = task_parts[1]
            target_name = task_parts[3]
            return f"Pick up the {object_name}, heat it, and place it in/on the {target_name}"

        elif task_type == "pick_two_obj_and_place":
            object_name = task_parts[1]
            target_name = task_parts[3]
            return f"Pick up two {object_name}s and place them in/on the {target_name}"

        elif task_type == "pick_and_place_with_movable_recep":
            object_name = task_parts[1]
            receptacle = task_parts[2]
            target_name = task_parts[3]
            return f"Pick up the {object_name} with the {receptacle} and place them in/on the {target_name}"

        else:
            return task_name

    def generate_environment_description(
        self, task: Dict[str, str], agent: Agent
    ) -> str:
        """
        Generate a detailed environment description for a task.

        Args:
            task (Dict): Task dictionary with task details
            agent (Agent): Agent to use for generating the environment description

        Returns:
            String describing the task environment
        """
        task_description = self.extract_task_description(task)
        task_type = task["task_type"]

        print(f"GENERATING ENVIRONMENT DESCRIPTION FOR: {task_description}")

        prompt = f"""
        Generate a detailed description of the environment for this ALFWorld task: "{task_description}"
        
        You need to provide a realistic and detailed description of the room(s), objects, and their locations.
        Include details about:
        - What rooms are present (kitchen, living room, bathroom, etc.)
        - Key objects and furniture in those rooms
        - Where the objects mentioned in the task can be found
        - Any containers or appliances that might be needed for the task
        - Spatial relationships between objects
        
        The task type is: {task_type}
        
        Be detailed but concise. This will serve as the environment context for an agent trying to complete the task.
        """

        completion_params = {
            "model": agent.deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if agent.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 500
        else:
            completion_params["max_tokens"] = 500

        completion = agent.client.chat.completions.create(**completion_params)
        environment_description = json.loads(completion.to_json())["choices"][0][
            "message"
        ]["content"]

        print("ENVIRONMENT DESCRIPTION GENERATED")
        return environment_description

    def evaluate_agent_actions(
        self, task_description: str, environment: str, actions: List[str], agent: Agent
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent's actions successfully complete the task.

        Args:
            task_description (str): Description of the task
            environment (str): Environment description
            actions (List[str]): List of actions taken by the agent
            agent (Agent): Agent to use for evaluation

        Returns:
            Dict with evaluation results (success, explanation)
        """
        print("EVALUATING AGENT ACTIONS EXTERNALLY")

        prompt = f"""
        You are evaluating whether a sequence of actions successfully completes a household task.
        
        TASK DESCRIPTION:
        {task_description}
        
        ENVIRONMENT:
        {environment}
        
        ACTIONS TAKEN:
        {json.dumps(actions, indent=2)}
        
        Evaluate whether these actions would successfully complete the task in the given environment.
        Consider:
        1. If all required steps are present
        2. If actions are in the correct order
        3. If the actions make reasonable assumptions about the environment
        4. If any critical actions are missing
        
        Return your evaluation in this JSON format:
        {{
            "success": true/false,
            "explanation": "detailed explanation of why the actions succeed or fail"
        }}

        The available actions are:
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

        So stick to the actions above and do not look for or evaluate based on any other actions.
        """

        completion_params = {
            "model": agent.deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "stream": False,
        }

        # Use the appropriate parameter based on model
        if agent.deployment == "o3-mini":
            completion_params["max_completion_tokens"] = 600
        else:
            completion_params["max_tokens"] = 600

        completion = agent.client.chat.completions.create(**completion_params)
        raw_response = json.loads(completion.to_json())["choices"][0]["message"][
            "content"
        ]

        try:
            evaluation_result = parse_json_from_response(raw_response)
            print(f"EVALUATION RESULT: {evaluation_result}")
            return evaluation_result
        except ValueError as e:
            print(f"Error parsing evaluation result: {e}")
            return {
                "success": False,
                "explanation": f"Failed to parse evaluation: {str(e)}",
            }

    def eval_tasks(
        self,
        tasks: List[Dict[str, str]],
        use_react=True,
        use_gpt4o=True,
        use_o3mini=True,
    ) -> Dict[str, Any]:
        """
        Evaluate the tasks by running them through the agent and comparing to expected outcomes.

        Args:
            tasks: List of task dictionaries to evaluate
            use_react: Whether to evaluate using the React agent
            use_gpt4o: Whether to evaluate using the GPT-4o direct agent
            use_o3mini: Whether to evaluate using the o3-mini direct agent

        Returns:
            Dictionary with evaluation results
        """
        agent_gpt4o = Agent("gpt-4o")
        agent_o3mini = Agent("o3-mini")
        evaluation_results = {
            "react_results": {
                "successful_tasks": 0,
                "task_results": [],
            },
            "direct_results": {
                "successful_tasks": 0,
                "task_results": [],
            },
            "o3mini_results": {
                "successful_tasks": 0,
                "task_results": [],
            },
            "evaluation_progress": {
                "current_task": 0,
                "total_tasks": len(tasks),
                "current_agent": "",
                "thinking_round": 0,
                "status": "",
                "task_type": "",
                "task_description": "",
            },
        }

        for index, task in enumerate(tasks):
            print(f"\nEVALUATING TASK {index + 1}")
            task_description = self.extract_task_description(task)
            print(f"TASK: {task_description}")

            # Generate environment description for the task
            environment_description = self.generate_environment_description(
                task, agent_gpt4o
            )

            # Update progress information
            evaluation_results["evaluation_progress"]["current_task"] = index + 1
            evaluation_results["evaluation_progress"]["status"] = "evaluating"
            evaluation_results["evaluation_progress"]["task_type"] = task["task_type"]
            evaluation_results["evaluation_progress"][
                "task_description"
            ] = task_description

            # Direct GPT-4o agent evaluation if selected
            if use_gpt4o:
                evaluation_results["evaluation_progress"][
                    "current_agent"
                ] = "Direct Agent (GPT-4o)"
                try:
                    print("DIRECT GPT-4O AGENT EVALUATION:")

                    # Include environment description in the prompt
                    enhanced_task = f"""
                    Task: {task_description}
                    
                    Environment Description:
                    {environment_description}
                    
                    Based on this task and environment, provide a sequence of actions that would complete the task.
                    """

                    raw_response = agent_gpt4o.alfworld_chat_direct(enhanced_task)
                    direct_response = parse_json_from_response(
                        raw_response["choices"][0]["message"]["content"]
                    )

                    direct_actions = direct_response.get("actions", [])
                    direct_reasoning = direct_response.get("reasoning", "")

                    print(f"\nDIRECT GPT-4O ACTIONS: {direct_actions}")
                    print(f"REASONING: {direct_reasoning}\n")

                    # External evaluation of actions
                    evaluation_result = self.evaluate_agent_actions(
                        task_description,
                        environment_description,
                        direct_actions,
                        agent_gpt4o,
                    )

                    direct_success = evaluation_result.get("success", False)
                    direct_explanation = evaluation_result.get("explanation", "")

                    if direct_success:
                        print("✅ DIRECT GPT-4O TASK EVALUATION: SUCCESSFUL")
                        evaluation_results["direct_results"]["successful_tasks"] += 1
                    else:
                        print("❌ DIRECT GPT-4O TASK EVALUATION: FAILED")

                    evaluation_results["direct_results"]["task_results"].append(
                        {
                            "task_type": task["task_type"],
                            "task_description": task_description,
                            "environment": environment_description,
                            "actions": direct_actions,
                            "reasoning": direct_reasoning,
                            "success": direct_success,
                            "evaluation_explanation": direct_explanation,
                        }
                    )
                except Exception as e:
                    print(f"Error in direct GPT-4o agent: {e}")
                    evaluation_results["direct_results"]["task_results"].append(
                        {
                            "task_type": task["task_type"],
                            "task_description": task_description,
                            "environment": environment_description,
                            "actions": [],
                            "reasoning": str(e),
                            "success": False,
                            "evaluation_explanation": f"Error: {str(e)}",
                        }
                    )

            # Direct o3-mini agent evaluation if selected
            if use_o3mini:
                evaluation_results["evaluation_progress"][
                    "current_agent"
                ] = "Direct Agent (o3-mini)"
                try:
                    print("DIRECT O3-MINI AGENT EVALUATION:")

                    # Include environment description in the prompt
                    enhanced_task = f"""
                    Task: {task_description}
                    
                    Environment Description:
                    {environment_description}
                    
                    Based on this task and environment, provide a sequence of actions that would complete the task.
                    """

                    raw_response = agent_o3mini.alfworld_chat_direct(enhanced_task)
                    o3mini_response = parse_json_from_response(
                        raw_response["choices"][0]["message"]["content"]
                    )

                    o3mini_actions = o3mini_response.get("actions", [])
                    o3mini_reasoning = o3mini_response.get("reasoning", "")

                    print(f"\nDIRECT O3-MINI ACTIONS: {o3mini_actions}")
                    print(f"REASONING: {o3mini_reasoning}\n")

                    # External evaluation of actions
                    evaluation_result = self.evaluate_agent_actions(
                        task_description,
                        environment_description,
                        o3mini_actions,
                        agent_gpt4o,  # Using GPT-4o for evaluation for consistency
                    )

                    o3mini_success = evaluation_result.get("success", False)
                    o3mini_explanation = evaluation_result.get("explanation", "")

                    if o3mini_success:
                        print("✅ DIRECT O3-MINI TASK EVALUATION: SUCCESSFUL")
                        evaluation_results["o3mini_results"]["successful_tasks"] += 1
                    else:
                        print("❌ DIRECT O3-MINI TASK EVALUATION: FAILED")

                    evaluation_results["o3mini_results"]["task_results"].append(
                        {
                            "task_type": task["task_type"],
                            "task_description": task_description,
                            "environment": environment_description,
                            "actions": o3mini_actions,
                            "reasoning": o3mini_reasoning,
                            "success": o3mini_success,
                            "evaluation_explanation": o3mini_explanation,
                        }
                    )
                except Exception as e:
                    print(f"Error in direct o3-mini agent: {e}")
                    evaluation_results["o3mini_results"]["task_results"].append(
                        {
                            "task_type": task["task_type"],
                            "task_description": task_description,
                            "environment": environment_description,
                            "actions": [],
                            "reasoning": str(e),
                            "success": False,
                            "evaluation_explanation": f"Error: {str(e)}",
                        }
                    )

            # React agent evaluation (using GPT-4o) if selected
            if use_react:
                evaluation_results["evaluation_progress"][
                    "current_agent"
                ] = "React Agent"
                print("\nREACT AGENT EVALUATION:")

                # Include environment description in the prompt
                enhanced_task = f"""
                Task: {task_description}
                
                Environment Description:
                {environment_description}
                
                Complete this task by thinking and taking actions in the environment.
                """

                message = enhanced_task
                react_verification_result = None
                all_actions = []

                for num in range(1, 8):
                    # Update thinking round for UI feedback
                    evaluation_results["evaluation_progress"]["thinking_round"] = num

                    try:
                        raw_response = agent_gpt4o.alfworld_chat_react(message)
                        raw_response = raw_response["choices"][0]["message"]["content"]
                        print("PARSING RESPONSE")
                        response = parse_json_from_response(raw_response)

                        if "thinking" in response.keys():
                            print(f"THINKING ROUND {num}: {response['thinking']}")
                            action = response["action"]
                            print(f"ACTION ROUND {num} : {action}\n")
                            all_actions.append(action)

                            # Get observation from executing action in environment
                            observation = agent_gpt4o.alfworld_observation_agent(action)
                            message = f"{message}\n{response}\n{observation}"
                            continue

                        elif "success" in response.keys():
                            reasoning = response.get("reasoning", "")
                            print(f"REASONING: {reasoning}\n")

                            # External evaluation of actions
                            evaluation_result = self.evaluate_agent_actions(
                                task_description,
                                environment_description,
                                all_actions,
                                agent_gpt4o,
                            )

                            react_success = evaluation_result.get("success", False)
                            react_explanation = evaluation_result.get("explanation", "")

                            if react_success:
                                print("✅ REACT TASK EVALUATION: SUCCESSFUL")
                                evaluation_results["react_results"][
                                    "successful_tasks"
                                ] += 1
                            else:
                                print("❌ REACT TASK EVALUATION: FAILED")

                            # Update progress information - completed
                            evaluation_results["evaluation_progress"][
                                "status"
                            ] = "completed"

                            evaluation_results["react_results"]["task_results"].append(
                                {
                                    "task_type": task["task_type"],
                                    "task_description": task_description,
                                    "environment": environment_description,
                                    "actions": all_actions,
                                    "reasoning": reasoning,
                                    "success": react_success,
                                    "evaluation_explanation": react_explanation,
                                }
                            )
                            react_verification_result = True
                            break

                    except Exception as e:
                        print(
                            f"Error processing task with React agent {index + 1}: {e}"
                        )
                        evaluation_results["react_results"]["task_results"].append(
                            {
                                "task_type": task["task_type"],
                                "task_description": task_description,
                                "environment": environment_description,
                                "actions": all_actions,
                                "reasoning": f"Error: {str(e)}",
                                "success": False,
                                "evaluation_explanation": f"Error: {str(e)}",
                            }
                        )
                        evaluation_results["evaluation_progress"]["status"] = "error"
                        react_verification_result = False
                        break

                # If we went through all rounds without getting a success result
                if react_verification_result is None:
                    print("❌ No result produced after maximum rounds with React agent")

                    # External evaluation of actions collected so far
                    if all_actions:
                        evaluation_result = self.evaluate_agent_actions(
                            task_description,
                            environment_description,
                            all_actions,
                            agent_gpt4o,
                        )

                        react_success = evaluation_result.get("success", False)
                        react_explanation = evaluation_result.get("explanation", "")
                    else:
                        react_success = False
                        react_explanation = "No actions were produced by the agent"

                    evaluation_results["evaluation_progress"]["status"] = "failed"
                    evaluation_results["react_results"]["task_results"].append(
                        {
                            "task_type": task["task_type"],
                            "task_description": task_description,
                            "environment": environment_description,
                            "actions": all_actions,
                            "reasoning": "No result produced after maximum rounds",
                            "success": react_success,
                            "evaluation_explanation": react_explanation,
                        }
                    )

        return evaluation_results


if __name__ == "__main__":
    # Command-line execution path
    dataset_path = "datasets\\alfworld"
    print("ENTER NUMBER OF TASKS TO RETRIEVE (0 FOR ALL):")
    num_tasks = int(input())

    try:
        print(f"Attempting to load dataset from: {dataset_path}")
        alfworld_eval = ALFWorldEval(dataset_path)
        alfworld_eval.load_alfworld_dataset()
        print("Obtain tasks to evaluate")
        tasks_to_evaluate = alfworld_eval.get_tasks(num_tasks)
        print("starting evaluation")
        result = alfworld_eval.eval_tasks(tasks_to_evaluate)
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
