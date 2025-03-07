import re
import json
import numpy as np
import torch

from vlmdrive import VLMDRIVE_REGISTRY
from vlmdrive.vlm.vlm_planner_base import VLMPlannerBase


@VLMDRIVE_REGISTRY.register
class VLMPlannerWaypoint(VLMPlannerBase):
    
    def _result_to_prediction_dict(self, result):
        """
        Convert the predicted waypoints into a dictionary with keys
        waypoints.
        
        Args:
            result (torch.Tensor): Tensor of shape (N, 2) containing waypoints.
            
        Returns:
            dict: Dictionary with keys 'target_speed', 'curvature', and 'dt'.
        """
        
        predicted_result = {
            'waypoints': result
        }
        return predicted_result
   
   
    def _get_ego_history(self, perception_memory_bank, agent_idx):
        """
        Build a JSON string containing the ego vehicle's historical relative waypoints for each time frame in the perception_memory_bank.

        The current frame's position is used as a reference for computing relative waypoints.

        Args:
            perception_memory_bank (list[dict]): List of perception records.
            agent_idx (int): Index of the ego vehicle in the perception data.

        Returns:
            str: A JSON string with the structure:
                {
                  "ego_history": [
                    {"timestamp": ..., "waypoints": [x, y]},
                    ...
                  ]
                }
        """
        ego_history_list = []

        # Current pose is used as reference for relative waypoints
        curr_pose = perception_memory_bank[-1]['detmap_pose'][agent_idx].cpu().numpy()
        curr_x, curr_y, _ = float(curr_pose[0]), float(curr_pose[1]), float(curr_pose[2])

        # Process each frame in the history (excluding the current frame)
        for frame_data in perception_memory_bank[:-1]:
            timestamp = frame_data['timestamp']
            # Extract ego pose: [x, y, yaw]
            ego_pose = frame_data['detmap_pose'][agent_idx].cpu().numpy()
            x, y, _ = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])

            record = {
                "timestamp": timestamp,
                # Compute relative waypoint with respect to the current frame
                "waypoints": [x - curr_x, y - curr_y]
            }
            ego_history_list.append(record)

        # Wrap the history list into a JSON structure and convert it to a string
        ego_history_json = {"ego_history": ego_history_list}
        return json.dumps(ego_history_json, indent=2)

        
    def _postprocess_result(self, result):
        """convert the predicted string-format waypoints into scalars and attach to result

        Args:
            result (str): string-format waypoints of waypoints

        Returns:
            dict: result with np.array waypoints
        """

        # 1) Extract JSON substring from the result using regex
        json_pattern = re.compile(
            r"```json\s*([\s\S]*?)\s*```|\{[\s\S]*\}",
            re.MULTILINE
        )
        json_str = None

        try:
            json_match = json_pattern.search(result)
            if json_match:
                # Use group(1) if available; otherwise, use the entire match
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            else:
                raise ValueError("No JSON content found in the model output.")
        except Exception as e:
            raise ValueError(f"Failed to locate JSON via regex: {str(e)}")

        # 2) Load the JSON structure
        try:
            json_result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        # 3) Retrieve the waypoints from the JSON
        if "predicted_waypoints" not in json_result:
            raise ValueError("JSON does not contain 'predicted_waypoints' key.")

        waypoint_list = json_result["predicted_waypoints"]
        if not isinstance(waypoint_list, list):
            raise ValueError("The 'predicted_waypoints' should be a list of pairs.")
        
        waypoints = torch.tensor(waypoint_list, dtype=torch.float32)
        return waypoints

