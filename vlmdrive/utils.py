import base64
import requests
import time
import random
import io
import base64
from math import atan2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid
import re
import json
from typing import Any, Dict, Optional
import asyncio
import concurrent.futures
from collections import defaultdict, deque

random.seed(42)


import sys
import io
import os
import time
import atexit
from pathlib import Path
from typing import Dict, Iterable, Optional
import asyncio, functools
from pathlib import Path

try:
    _to_thread = asyncio.to_thread  # 3.9+
except AttributeError:              # 3.8 fallback
    async def _to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

_init_lock = asyncio.Lock()

# ----------------------------
# Core: timestamped Logger
# ----------------------------
class Logger(io.TextIOBase):
    """
    A simple text stream that mirrors output to terminal AND appends to a file.
    - Terminal: prints message as-is
    - File: writes the same message, optionally prefixed with a timestamp
    """
    def __init__(
        self,
        file_name: str,
        stream=sys.stdout,
        with_ts_in_file: bool = True,
        echo_to_terminal: bool = True,
    ):
        self._terminal = stream
        self._file = open(file_name, "a", buffering=1, encoding="utf-8")
        self._with_ts = with_ts_in_file
        self._echo = echo_to_terminal

    # print(...) ultimately calls stream.write(str) -> we implement write
    def write(self, message: str):
        if not message:
            return
        if self._echo:
            # Terminal shows the message exactly as given (no timestamp)
            self._terminal.write(message)

        # File version (optionally with timestamp)
        if self._with_ts:
            ts = "" if message == "\n" else time.strftime("%Y-%m-%d %H:%M:%S")
            self._file.write((ts + "  " if ts else "") + message)
        else:
            self._file.write(message)

    def flush(self):
        try:
            self._file.flush()
        finally:
            if self._echo:
                try:
                    self._terminal.flush()
                except Exception:
                    pass

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

# Keep track so we can close all on process exit
_all_loggers = []

def _register_logger(lg: Logger):
    _all_loggers.append(lg)

def _close_all():
    for lg in _all_loggers:
        try:
            lg.close()
        except Exception:
            pass

atexit.register(_close_all)

# ----------------------------
# Registry: setup & accessors
# ----------------------------
_logger_map: Dict[str, Logger] = {}
_other_logger: Optional[Logger] = None
_initialized = False
_log_dir_cache: Optional[str] = None

def setup(agents: Iterable[str], log_dir: str = "logs"):
    """
    Initialize the registry ONCE per process.
    - Creates one file per agent (e.g., logs/car1.log)
    - Creates 'other.log' for messages without a known agent
    Subsequent calls are idempotent; new agents will be added if needed.
    """
    global _initialized, _other_logger, _log_dir_cache

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _log_dir_cache = log_dir

    # Create/ensure the shared "other" logger
    if _other_logger is None:
        other_path = os.path.join(log_dir, "other.log")
        _other = Logger(other_path)
        _register_logger(_other)
        # assign after creation to avoid partial state on exceptions
        globals()["_other_logger"] = _other

    # Create/ensure each agent logger
    for a in agents:
        if a not in _logger_map:
            path = os.path.join(log_dir, f"{a}.log")
            lg = Logger(path)
            _logger_map[a] = lg
            _register_logger(lg)

    _initialized = True

def get_logger(agent: Optional[str] = None) -> Logger:
    """
    Get the logger for a given agent.
    - If agent is None or unknown, returns the 'other' logger.
    - If setup() was never called, this will lazily create logs/other.log
      so that print(..., file=get_logger()) still works safely.
    """
    global _other_logger, _log_dir_cache
    if agent and agent in _logger_map:
        return _logger_map[agent]

    if _other_logger is None:
        # Lazy, defensive init: make sure we at least have logs/other.log
        default_dir = _log_dir_cache or "logs"
        setup(agents=[], log_dir=default_dir)
    return _other_logger  # type: ignore

async def get_logger_async(agent: Optional[str] = None) -> Logger:
    """
    Async variant of get_logger() with concurrency safety:
    - Returns existing agent logger if present.
    - Lazily initializes 'other.log' if registry not yet set up.
    - If an unknown agent name is provided, ensures that agent's logger exists.
    """
    global _other_logger, _log_dir_cache   # <--- move here

    # Fast paths without locking when possible
    if agent and agent in _logger_map:
        return _logger_map[agent]
    if _other_logger is not None and (agent is None or agent not in _logger_map):
        return _other_logger  # type: ignore

    async with _init_lock:
        if agent and agent in _logger_map:
            return _logger_map[agent]

        if _other_logger is None:
            default_dir = _log_dir_cache or "logs"
            await _to_thread(setup, agents=[], log_dir=default_dir)

        if agent and agent not in _logger_map:
            await _to_thread(ensure_agent, agent)

        return _logger_map.get(agent, _other_logger)  # type: ignore

def logger_map() -> Dict[str, Logger]:
    """
    Return the internal map {agent_name: Logger}. Read-only usage recommended.
    If setup() wasn't called, ensure we have a default 'other.log' ready.
    """
    if _other_logger is None:
        setup(agents=[], log_dir=_log_dir_cache or "logs")
    return _logger_map

def ensure_agent(agent: str):
    """
    Ensure a logger exists for the given agent. Safe to call at runtime when
    new agents appear dynamically.
    """
    global _log_dir_cache
    if agent in _logger_map:
        return
    # If setup() never ran, choose a default dir
    if _log_dir_cache is None:
        _log_dir_cache = "logs"
    Path(_log_dir_cache).mkdir(parents=True, exist_ok=True)
    path = os.path.join(_log_dir_cache, f"{agent}.log")
    lg = Logger(path)
    _logger_map[agent] = lg
    _register_logger(lg)



KEY = "<your-api-key>"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4(question, api_key=None, image_path=None, proxy='openai', sys_message=None):

    if proxy == "ohmygpt":
        request_url = "https://aigptx.top/v1/chat/completions"
    elif proxy == "openai":
        request_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    if image_path is not None:
        base64_image = encode_image(image_path)
        if sys_message is not None:
            params = {
                "messages": [
                    {
                    "role": "system", 
                    "content": sys_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }
        else:

            params = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": 'gpt-4o-mini-2024-07-18',
                "temperature": 0.0
            }
    else:
        if sys_message is not None:
            params = {
                "messages": [

                    {
                        "role": "system", 
                        "content": sys_message
                    },
                    {
                        "role": 'user',
                        "content": question
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }
        else:
            params = {
                "messages": [
                    {
                        "role": 'user',
                        "content": question
                    }
                ],
                "model": 'gpt-4o',
                "temperature": 0.0
            }


    received = False
    while not received:
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=params,
                stream=False
            )
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            received = True
        except:
            time.sleep(1)
    return res_content


def PlotBase64Image(image: str):
    i = base64.b64decode(image)
    i = io.BytesIO(i)
    i = mpimg.imread(i, format='JPG')

    plt.imshow(i, interpolation='nearest')
    plt.show()



def TransformPoint(point, transform):
    """ Transform a 3D point using a transformation matrix. """
    if isinstance(point, list):
        point = np.array(point)

    if point.shape[-1] == 3:
        point = np.append(point, 1)
    transformed_point = transform @ point
    return transformed_point[:3]

def FormTransformationMatrix(translation, rotation):
    """ Create a transformation matrix from translation and rotation (as a quaternion). """
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def ProjectEgoToImage(points_3d: np.array, K):
    """ Project 3D points to 2D using camera intrinsic matrix K. """
    # Filter out points that are behind the camera
    points_3d = points_3d[points_3d[:, 2] > 0]

    # Project the remaining points
    points_2d = np.dot(K, points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2][:, np.newaxis]  # Normalize by depth
    return points_2d

def ProjectWorldToImage(points3d_world: list, cam_to_ego, ego_to_world):
    # Plot the waypoints.

    T_ego_global = FormTransformationMatrix(ego_to_world['translation'], Quaternion(ego_to_world['rotation']))
    T_cam_ego = FormTransformationMatrix(cam_to_ego['translation'], Quaternion(cam_to_ego['rotation']))
    T_cam_global = T_ego_global @ T_cam_ego
    T_global_cam = np.linalg.inv(T_cam_global)

    points3d_cam = [TransformPoint(point, T_global_cam) for point in points3d_world]

    points3d_img = ProjectEgoToImage(np.array(points3d_cam), cam_to_ego['camera_intrinsic'])

    return points3d_img


def OffsetTrajectory3D(points, offset_distance):
    """
    Offsets a 3D trajectory by a specified distance normal to the trajectory.

    Parameters:
        points (np.ndarray): n x 3 array representing the 3D trajectory (x, y, z).
        offset_distance (float): Distance to offset the trajectory.

    Returns:
        np.ndarray: Offset trajectory as an n x 3 array.
    """
    # Compute differences to find tangent vectors
    tangents = np.gradient(points, axis=0)  # Approximate tangents
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)  # Normalize tangents

    # Reference vector for normal plane computation (e.g., z-axis)
    reference_vector = np.array([0, 0, 1])

    # Compute normal vectors via cross product
    normals = np.cross(tangents, reference_vector)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize normals

    # Compute offset points
    offset_points = points + offset_distance * normals

    return offset_points

def OverlayTrajectory(img, points3d_world: list, cam_to_ego, ego_to_world, color=(0, 0, 255), args=None):

    # Construct left/right boundaries.
    points3d_left_world = OffsetTrajectory3D(np.array(points3d_world), -1.73 / 2)
    points3d_right_world = OffsetTrajectory3D(np.array(points3d_world), 1.73 / 2)

    # Project the waypoints to the image.
    points3d_img = ProjectWorldToImage(points3d_world, cam_to_ego, ego_to_world)
    points3d_left_img = ProjectWorldToImage(points3d_left_world.tolist(), cam_to_ego, ego_to_world)
    points3d_right_img = ProjectWorldToImage(points3d_right_world.tolist(), cam_to_ego, ego_to_world)

    if args.plot:
        # Overlay the waypoints on the image.
        for i in range(len(points3d_img) - 1):
            cv.circle(img, tuple(points3d_img[i].astype(int)), radius=6, color=color, thickness=-1)

        # # Draw lines.
        # for i in range(len(points3d_img) - 1):
        #     cv.line(img, tuple(points3d_img[i].astype(int)), tuple(points3d_img[i+1].astype(int)), color, 2)

    # Draw sweep area polygon between the boundaries.
    frame = np.zeros_like(img)
    polygon = np.vstack((np.array(points3d_left_img), np.array(points3d_right_img)[::-1])).astype(np.int32)
    check_flag = False
    if polygon.size == 0:
        check_flag = True
        return check_flag
    if args.plot:
        cv.fillPoly(frame, [polygon], color=color)  # Green polygon
        mask = frame.astype(bool)
        img[mask] = cv.addWeighted(img, 0.5, frame, 0.5, 0)[mask]
    return check_flag



def EstimateCurvatureFromTrajectory(traj):
    traj = traj[:, :2]

    # Initialize curvature array
    curvature = np.zeros(len(traj))

    # Compute curvature at each point (excluding the first and last)
    for i in range(1, len(traj) - 1):
        x1, y1 = traj[i - 1][0], traj[i - 1][1]
        x2, y2 = traj[i][0], traj[i][1]
        x3, y3 = traj[i + 1][0], traj[i + 1][1]

        # Compute side lengths
        L1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        L2 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        L3 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        # Compute triangle area
        area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Compute curvature
        if L1 > 0 and L2 > 0 and L3 > 0:  # Avoid division by zero
            curvature[i] = 4 * area / (L1 * L2 * L3)

    curvature[0] = curvature[1]  # Set the first curvature to the second
    curvature[-1] = curvature[-2]  # Set the last curvature to the second-to-last

    return curvature

def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    t = np.linspace(0, time_span, time_span)  # Time vector

    # Initial conditions
    x0, y0 = initial_position[0], initial_position[1]  # Starting position
    theta0 = initial_heading  # Initial orientation (radians)

    # Integrate to compute heading (theta)
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=theta0)
    theta[1:] += theta0

    # Compute velocity components
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)

    # Integrate to compute trajectory
    x = cumulative_trapezoid(v_x, t, initial=x0)
    y = cumulative_trapezoid(v_y, t, initial=y0)

    x[1:] += x0
    y[1:] += y0

    return np.stack((x, y), axis=1)

def WriteImageSequenceToVideo(cam_images_sequence: list, filename):
    assert len(cam_images_sequence) >= 1, "No images to write to video."
    # Save the image sequence as video
    # Define the codec and initialize the VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv.VideoWriter(f"{filename}.mp4", fourcc, fps=2,
                                   frameSize=(cam_images_sequence[0].shape[1], cam_images_sequence[0].shape[0]))

    for img in cam_images_sequence:
        video_writer.write(img)

    # Release the video writer
    video_writer.release()
    
    

def _extract_balanced_json(text: str) -> Optional[str]:
    """Prefer fenced ```json blocks; else scan for first balanced {...} honoring quotes/escapes."""
    # 1) Fenced block (```json ... ```), case-insensitive
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)

    # 2) Balanced-brace scan that ignores braces inside quoted strings
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str: Optional[str] = None  # None or the quote char (' or ")
    escape = False

    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _load_json_loose(s: str) -> Any:
    """Strict json first; on failure, repair common LLM artifacts and try again."""
    try:
        return json.loads(s)
    except Exception:
        pass

    repaired = s

    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*(?=[}\]])", "", repaired)

    # Python-ish literals -> JSON
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)

    # Bare YES/NO as values -> quoted (case-insensitive)
    repaired = re.sub(r'(:\s*)(YES|Yes|yes)(\s*[}\],])', r'\1"YES"\3', repaired)
    repaired = re.sub(r'(:\s*)(NO|No|no)(\s*[}\],])', r'\1"NO"\3', repaired)

    # If the text appears to use only single quotes for strings, convert to double.
    # Heuristic: only do this if there are no existing double-quoted strings.
    if '"' not in repaired and "'" in repaired:
        repaired = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', repaired)

    # Try again
    return json.loads(repaired)


def str_parse_json(input_str: str) -> Dict[str, Any]:
    """
    Parse a string containing JSON-like content into a Python dictionary.

    Behavior:
      1) Prefer fenced ```json blocks.
      2) Otherwise, extract the first balanced {...} (quotes/escapes respected).
      3) Try strict JSON parsing; if that fails, apply small repairs and parse again.

    Returns:
        dict: Parsed dictionary from the input string. If no JSON is found, returns {}.
    Raises:
        ValueError: If JSON-like content is found but cannot be parsed even after repairs.
    """
    # Extract
    json_str = _extract_balanced_json(input_str)
    if json_str is None:
        return {}

    # Parse (strict, then loose)
    try:
        return _load_json_loose(json_str)
    except Exception as e:
        # Provide context but avoid dumping massive strings
        preview = json_str[:200].replace("\n", " ")
        raise ValueError(f"Failed to parse JSON (preview: {preview!r}...): {e}") from e
    

def run_coro_blocking(coro):
    """Run an async coroutine from sync code.
    - If no loop: asyncio.run
    - If already in a running loop (e.g., Jupyter/FastAPI): run in a worker thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # no running loop
        return asyncio.run(coro)

    # already in a running loop -> use a fresh loop in a worker thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()
    
    
import os
import json
import tempfile
import threading
from pathlib import Path
from collections import defaultdict, deque

class StatManager:
    def __init__(self, pid=None, window: int = 100):
        # 1) Safe RESULT_ROOT with default
        self.root = Path(os.environ.get('RESULT_ROOT', '.')) / "time_stats"
        self.pid = pid
        # self.log_dir = root / f"time_stats_pid_{pid}"
        # self.log_dir.mkdir(parents=True, exist_ok=True)

        self._window = int(window)
        self.stats_time = defaultdict(lambda: deque(maxlen=self._window))
        # 3) Concurrency guard
        self._lock = threading.Lock()

    def update_time(self, key: str, value):
        # 5) Normalize types defensively
        try:
            v = float(value)
        except Exception:
            return  # or raise/log if you prefer strictness
        with self._lock:
            self.stats_time[key].append(v)

    def log_time_stats(self, run_time_idx=0):
        with self._lock:
            avg_stat = {}
            for key, values in self.stats_time.items():
                if values:
                    avg_stat[key] = float(sum(values) / len(values))
                    
        self.log_dir = self.root / run_time_idx / f"pid_{self.pid}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 4) Atomic write
        tmp = tempfile.NamedTemporaryFile(
            mode="w", delete=False, dir=self.log_dir, prefix="time_stats_", suffix=".json"
        )
        try:
            json.dump(avg_stat, tmp, indent=4)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.log_dir / "time_stats.json")
        except Exception:
            try:
                tmp.close()
                os.unlink(tmp.name)
            except Exception:
                pass
            raise