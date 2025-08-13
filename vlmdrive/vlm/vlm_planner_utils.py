# vlm_planner_utils.py
from typing import Dict, Any, List, Optional
import logging
import numpy as np
import torch
from pathlib import Path
from vlmdrive.vlm_api_helper import VLMAPIHelper

_logger = logging.getLogger(__name__)

STAGES = ['scene', 'object', 'intention', 'target', 'comb']


def _read_api_key(api_key: str) -> str:
    """Read API key; if it's a path, load from file, else return as-is."""
    try:
        p = Path(api_key)
        if p.exists() and p.is_file():
            _logger.info(f"API key loaded from {str(p)}")
            return p.read_text().strip()
    except Exception:
        pass
    _logger.info("API key loaded from direct input")
    return api_key


def configure_vlm_helpers(
    name: Any,
    api_model_name: Any,
    api_base_url: Any,
    api_key: Any,
    image_placeholder: str,
) -> Dict[str, VLMAPIHelper]:
    """
    Build VLM helpers for each stage based on `name` spec.
    - If `name` is a str containing 'api': one helper shared by all stages.
    - If `name` is a dict per stage: build a helper per stage.
    """
    helpers: Dict[str, VLMAPIHelper] = {}

    if isinstance(name, str):
        if "api" not in name:
            raise ValueError(f"Unsupported model path: {name}")
        final_key = _read_api_key(api_key)
        helper = VLMAPIHelper(
            api_key=final_key,
            api_base_url=api_base_url,
            api_model_name=api_model_name,
            image_placeholder=image_placeholder,
        )
        for s in STAGES:
            helpers[s] = helper
        return helpers

    if isinstance(name, dict):
        assert isinstance(api_model_name, dict), "api_model_name should be a dict"
        assert isinstance(api_base_url, dict), "api_base_url should be a dict"
        assert isinstance(api_key, dict), "api_key should be a dict"

        for stage, model_path in name.items():
            if "api" not in model_path:
                raise ValueError(f"Unsupported model path: {model_path}")
            final_key = _read_api_key(api_key[stage])
            helpers[stage] = VLMAPIHelper(
                api_key=final_key,
                api_base_url=api_base_url[stage],
                api_model_name=api_model_name[stage],
                image_placeholder=image_placeholder,
            )
        # For any stage not explicitly provided, fall back to one that exists (keeps old behavior robust)
        fallback = helpers.get('scene') or helpers.get('comb') or list(helpers.values())[0]
        for s in STAGES:
            helpers.setdefault(s, fallback)
        return helpers

    # Keep original error text semantics
    raise ValueError(f"model path: {name} should be a string or a list of strings, not {type(name)}")


def ensure_list(images):
    """Ensure images is a list (API expects list)."""
    return images if isinstance(images, list) else [images]


def get_ego_front_image(perception_memory_bank: List[dict], agent_idx: int):
    """Return the ego front RGB image of shape (H, W, 3)."""
    return perception_memory_bank[-1]['rgb_front'][agent_idx]


def get_all_objects(perception_memory_bank: List[dict]):
    """Return CAV detected objects with bounding boxes."""
    return perception_memory_bank[-1]["object_list"]


def get_collab_agent_images(
    perception_memory_bank: List[dict],
    model_config: dict,
    ego_idx: int
) -> Optional[Dict[int, np.ndarray]]:
    """Collect front images for all agents except ego, if image sharing is enabled."""
    if "image" not in model_config['collab']['sharing_modalities']:
        return None
    front_images = perception_memory_bank[-1]["rgb_front"]
    out = {}
    N = front_images.shape[0]
    for i in range(N):
        if i != ego_idx:
            out[i] = front_images[i]
    return out


def get_related_pos_with_direction(
    ego_pos,  # torch.Tensor | np.ndarray, shape (2,)
    ego_yaw: float,  # radians
    positions,  # torch.Tensor | np.ndarray, shape (N,2)
) -> np.ndarray:
    """Convert global positions to ego-centric coordinates aligned with ego forward."""
    if isinstance(ego_pos, torch.Tensor):
        ego_pos = ego_pos.detach().cpu().numpy()
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    relative_global_pos = positions - ego_pos  # (N, 2)

    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    rotation_matrix = np.array([
        [ sin_yaw,  cos_yaw],  # local x
        [-cos_yaw,  sin_yaw],  # local y
    ])
    return relative_global_pos @ rotation_matrix.T  # (N, 2)


def build_collab_agent_description(
    collab_agent_intent: List[dict],
    image_placeholder: str,
    with_image: bool = False,
) -> str:
    """
    Build description for collaborative agents.
    Each agent contributes a line with: idx, position (rounded), intent text, and an IMAGE_PLACEHOLDER.
    """
    description = ""
    for intent in collab_agent_intent:
        position = intent.get('position', '')

        # Convert to list and round for readability, preserving original prints
        if isinstance(position, torch.Tensor):
            position = position.detach().cpu().numpy()
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if isinstance(position, list):
            position = [round(float(coord), 5) for coord in position]

        description += (
            f"Agent {intent['idx']}, located at: {position}, "
            f"scene_description: {intent.get('scene_description', '')}, "
            f"object_description: {intent.get('object_description', '')}, "
            f"target_description: {intent.get('target_description', '')}, "
            f"intent description: {intent.get('intent_description', '')}, "
        )
        
        if with_image:
            description += f"Image: {image_placeholder}\n"
        
        print(f"Collaborative agent {intent['idx']} description: {description}")
    return description


def build_target_description(target_waypoint, prompt_template: dict, idx: int) -> str:
    """
    Build natural language target description: mirrors original logic exactly.
    """
    x = float(target_waypoint[0])
    y = -float(target_waypoint[1])
    tmpl = prompt_template["target_prompt_template"]["default"]

    x_direction = "right" if x > 0 else "left"
    y_direction = "front" if y > 0 else "back"
    x_distance = abs(round(x, 5))
    y_distance = abs(round(y, 5))

    prompt = tmpl.format(
        x_distance=x_distance,
        x_direction=x_direction,
        y_distance=y_distance,
        y_direction=y_direction
    )
    print(f"Target description Prompt: {prompt}, idx: {idx}")
    return prompt