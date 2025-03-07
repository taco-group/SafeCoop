import re
import json
import numpy as np
import torch

from vlmdrive import VLMDRIVE_REGISTRY
from vlmdrive.vlm.vlm_planner_base import VLMPlannerBase


@VLMDRIVE_REGISTRY.register
class VLMPlannerControl(VLMPlannerBase):
    
    def _result_to_prediction_dict(self, result):
        """
        Convert the predicted waypoints into a dictionary with keys
        waypoints.
        
        Args:
            result (torch.Tensor): Tensor of shape (N, 2) containing waypoints.
            
        Returns:
            dict: Dictionary with keys 'steering', 'throttle', and 'brake'.
        """
        
        predicted_result = {
            'steering': [s[0] for s in result],
            'throttle': [t[1] for t in result],
            'brake': [b[2] for b in result]
        }
        
        return predicted_result
   
   
    def _get_ego_history(self, perception_memory_bank, agent_idx):
        """
        Build a JSON string containing the ego vehicle's historical speed, curvature,
        and relative waypoints for each time frame in the perception_memory_bank.

        The current frame's position is used as a reference for computing relative waypoints.

        Args:
            perception_memory_bank (list[dict]): List of perception records.
            agent_idx (int): Index of the ego vehicle in the perception data.

        Returns:
            str: A JSON string with the structure:
                {
                  "ego_history": [
                    {"timestamp": ..., 
                    "speed": ..., 
                    "curvature": ..., 
                    "waypoints": [x, y],
                    "steering": ...,
                    "throttle": ...,
                    "brake": ...},
                    ...
                  ]
                }
        """
        ego_history_list = []
        prev_position = None
        prev_yaw = None
        dt = 0.5  # Time interval in seconds

        # Process each frame in the history (excluding the current frame)
        for frame_data in perception_memory_bank[:-1]:
            timestamp = frame_data['timestamp']
            # Extract ego pose: [x, y, yaw]
            ego_pose = frame_data['detmap_pose'][agent_idx].cpu().numpy()
            x, y, _ = float(ego_pose[0]), float(
                ego_pose[1]), float(ego_pose[2])
            yaw = frame_data['ego_yaw'][agent_idx]

            # Calculate speed as the derivative of position
            if prev_position is not None:
                dx = x - prev_position[0]
                dy = y - prev_position[1]
                speed = np.sqrt(dx * dx + dy * dy) / dt
            else:
                speed = 0.0

            # Calculate curvature as the change in yaw per unit distance
            if prev_yaw is not None:
                d_yaw = yaw - prev_yaw
                distance = max(1e-6, speed * dt)
                curvature = d_yaw / distance
            else:
                curvature = 0.0
                
            record = {
                "timestamp": timestamp,
                "speed": round(speed, 5),
                "curvature": round(curvature, 5),
            }
            record.update(frame_data['predicted_result_list'][agent_idx])
            
            ego_history_list.append(record)
            prev_position = (x, y)
            prev_yaw = yaw

        # Wrap the history list into a JSON structure and convert it to a string
        ego_history_json = {"ego_history": ego_history_list}
        return json.dumps(ego_history_json, indent=2)

    def _postprocess_result(self, result: str):
        """
        Parse and normalize the predicted control signals JSON output.

        Expected JSON format:
        {
            "predicted_control_signal": [
                [steering_1, throttle_1, brake_1],
                [steering_2, throttle_2, brake_2],
                ...
                [steering_5, throttle_5, brake_5]
            ]
        }

        Where:
        - steering: a value convertible to float (range [-1, 1])
        - throttle: a value convertible to float (range [0, 1])
        - brake: a boolean value, but may appear as various case strings ("true"/"false") or integer 0/1.

        This function extracts the JSON, validates each control signal entry, and normalizes:
        - steering and throttle to float
        - brake to boolean

        Args:
            result (str): String containing the JSON-formatted control signals.

        Returns:
            list: A list of normalized control signal entries.
        """
        import re
        import json

        # 1) Extract the JSON substring using regex
        json_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```|\{[\s\S]*\}", re.MULTILINE)
        json_match = json_pattern.search(result)
        if not json_match:
            raise ValueError("No JSON content found in the model output.")
        # Use captured group if available; otherwise, use the full match
        json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)

        # 2) Parse the JSON structure
        try:
            json_result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        # 3) Retrieve the control signals from the JSON
        if "predicted_control_signal" not in json_result:
            raise ValueError("JSON does not contain 'predicted_control_signal' key.")
        control_signals = json_result["predicted_control_signal"]
        if not isinstance(control_signals, list):
            raise ValueError("'predicted_control_signal' should be a list.")

        # 4) Validate and normalize each control signal entry
        for idx, entry in enumerate(control_signals):
            if not isinstance(entry, list):
                raise ValueError(f"Control signal at index {idx} is not a list.")
            if len(entry) != 3:
                raise ValueError(f"Control signal at index {idx} does not contain exactly 3 elements.")
            
            steering, throttle, brake = entry

            # Normalize steering: ensure convertible to float
            try:
                steering = float(steering)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Steering value at index {idx} cannot be converted to float: {e}")

            # Normalize throttle: ensure convertible to float
            try:
                throttle = float(throttle)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Throttle value at index {idx} cannot be converted to float: {e}")

            # Normalize brake: accept bool, string (any case), or integer 0/1
            if isinstance(brake, bool):
                pass  # Already a boolean
            elif isinstance(brake, str):
                lower_brake = brake.lower()
                if lower_brake == "true":
                    brake = True
                elif lower_brake == "false":
                    brake = False
                else:
                    raise ValueError(f"Brake value at index {idx} is an unrecognized string: {brake}")
            elif isinstance(brake, int):
                if brake in (0, 1):
                    brake = bool(brake)
                else:
                    raise ValueError(f"Brake value at index {idx} integer must be 0 or 1, got {brake}")
            else:
                raise ValueError(f"Brake value at index {idx} is of unsupported type: {type(brake)}")

            # Update the normalized entry in the control_signals list
            control_signals[idx] = [steering, throttle, brake]

        return control_signals