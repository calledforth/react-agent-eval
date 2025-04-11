import os
import json
import datetime
from typing import Dict, Any, List, Optional
import uuid


class HistoryManager:
    """
    Manager for storing and retrieving evaluation history.
    """

    def __init__(self, storage_dir: str = "evaluation_history"):
        """
        Initialize the history manager.

        Args:
            storage_dir: Directory to store evaluation results
        """
        self.storage_dir = storage_dir
        # Create directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def save_evaluation(
        self,
        eval_type: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save an evaluation result to persistent storage.

        Args:
            eval_type: Type of evaluation (hotpotqa, fever, alfworld)
            results: The evaluation results to store
            metadata: Additional metadata about the evaluation

        Returns:
            id: Unique ID for the saved evaluation
        """
        # Generate a unique ID for this evaluation
        eval_id = str(uuid.uuid4())

        # Create metadata if not provided
        if metadata is None:
            metadata = {}

        # Add timestamp and evaluation type
        timestamp = datetime.datetime.now().isoformat()
        metadata["timestamp"] = timestamp
        metadata["eval_type"] = eval_type

        # Create the complete record
        record = {"id": eval_id, "metadata": metadata, "results": results}

        # Create a filename with timestamp for easy sorting
        filename = f"{eval_type}_{timestamp.replace(':', '-').replace('.', '-')}_{eval_id[:8]}.json"
        filepath = os.path.join(self.storage_dir, filename)

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return eval_id

    def get_evaluation_history(
        self, eval_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation history, optionally filtered by type.

        Args:
            eval_type: Type of evaluations to retrieve (None for all)

        Returns:
            List of evaluation metadata records
        """
        history = []

        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    record = json.load(f)

                # Filter by type if specified
                if (
                    eval_type is None
                    or record["metadata"].get("eval_type") == eval_type
                ):
                    # Include basic metadata and ID, but not full results
                    history.append(
                        {
                            "id": record["id"],
                            "filename": filename,
                            "metadata": record["metadata"],
                            # Include summary metrics if available
                            "summary": self._extract_summary_metrics(
                                record["results"], record["metadata"].get("eval_type")
                            ),
                        }
                    )
            except Exception as e:
                print(f"Error loading history file {filename}: {str(e)}")

        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
        return history

    def get_evaluation_by_id(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific evaluation by ID.

        Args:
            eval_id: ID of the evaluation to retrieve

        Returns:
            The evaluation record or None if not found
        """
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    record = json.load(f)

                if record["id"] == eval_id:
                    return record
            except Exception as e:
                print(f"Error loading history file {filename}: {str(e)}")

        return None

    def _extract_summary_metrics(
        self, results: Dict[str, Any], eval_type: str
    ) -> Dict[str, Any]:
        """Extract key metrics from results based on evaluation type."""
        summary = {"react": {}, "direct": {}, "o3mini": {}}

        try:
            if eval_type == "hotpotqa":
                if "react_results" in results:
                    summary["react"] = {
                        "correct": results["react_results"].get("correct_answers", 0),
                        "total": len(
                            results["react_results"].get("question_answer_pairs", [])
                        ),
                    }
                if "direct_results" in results:
                    summary["direct"] = {
                        "correct": results["direct_results"].get("correct_answers", 0),
                        "total": len(
                            results["direct_results"].get("question_answer_pairs", [])
                        ),
                    }
                if "o3mini_results" in results:
                    summary["o3mini"] = {
                        "correct": results["o3mini_results"].get("correct_answers", 0),
                        "total": len(
                            results["o3mini_results"].get("question_answer_pairs", [])
                        ),
                    }

            elif eval_type == "fever":
                if "react_results" in results:
                    summary["react"] = {
                        "correct": results["react_results"].get(
                            "correct_verifications", 0
                        ),
                        "total": len(
                            results["react_results"].get("claim_verification_pairs", [])
                        ),
                    }
                if "direct_results" in results:
                    summary["direct"] = {
                        "correct": results["direct_results"].get(
                            "correct_verifications", 0
                        ),
                        "total": len(
                            results["direct_results"].get(
                                "claim_verification_pairs", []
                            )
                        ),
                    }
                if "o3mini_results" in results:
                    summary["o3mini"] = {
                        "correct": results["o3mini_results"].get(
                            "correct_verifications", 0
                        ),
                        "total": len(
                            results["o3mini_results"].get(
                                "claim_verification_pairs", []
                            )
                        ),
                    }

            elif eval_type == "alfworld":
                if "react_results" in results:
                    summary["react"] = {
                        "successful": results["react_results"].get(
                            "successful_tasks", 0
                        ),
                        "total": len(results["react_results"].get("task_results", [])),
                    }
                if "direct_results" in results:
                    summary["direct"] = {
                        "successful": results["direct_results"].get(
                            "successful_tasks", 0
                        ),
                        "total": len(results["direct_results"].get("task_results", [])),
                    }
                if "o3mini_results" in results:
                    summary["o3mini"] = {
                        "successful": results["o3mini_results"].get(
                            "successful_tasks", 0
                        ),
                        "total": len(results["o3mini_results"].get("task_results", [])),
                    }
        except Exception as e:
            print(f"Error extracting summary metrics: {str(e)}")

        return summary
