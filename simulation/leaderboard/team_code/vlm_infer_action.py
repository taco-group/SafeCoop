import os
import copy
import re
import io
import logging
import json
import numpy as np
import torch
import carla
import cv2
import math
import datetime
import pathlib
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from skimage.measure import block_reduce
import time
from typing import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from team_code.planner import RoutePlanner
import torch.nn.functional as F
import pygame
import queue
from copy import deepcopy

import pdb

from agents.navigation.local_planner import RoadOption
from team_code.eval_utils import turn_traffic_into_bbox_fast
from team_code.render_v2x import render, render_self_car, render_waypoints
from team_code.v2x_utils import (generate_relative_heatmap, 
				 generate_heatmap, generate_det_data,
				 get_yaw_angle, boxes_to_corners_3d, get_points_in_rotated_box_3d  # visibility related functions
				 )

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from opencood.tools import train_utils
from opencood.tools import train_utils, inference_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.visualization import vis_utils, my_vis, simple_vis_multiclass

####### Input: raw_data, N(actor)+M(RSU)
####### Output: actors action, N(actor)
####### Generate the action with the trained model.

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
os.environ["SDL_VIDEODRIVER"] = "dummy"

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    return collides, p1 + x[0] * v1

class DisplayInterface(object):
	def __init__(self):
		self._width = 2300
		self._height = 600
		self._surface = None

		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		self._display = pygame.display.set_mode(
			(self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
		)
		pygame.display.set_caption("V2X Agent")

	def run_interface(self, input_data):
		rgb = input_data['rgb']
		map = input_data['map']
		lidar = input_data['lidar']
		surface = np.zeros((600, 2300, 3),np.uint8)
		surface[:, :800] = rgb
		surface[:,800:1400] = lidar
		surface[:,1400:2000] = input_data['lidar_rsu']
		surface[:,2000:2300] = input_data['map']
		surface[:150,:200] = input_data['rgb_left']
		surface[:150, 600:800] = input_data['rgb_right']
		surface[:150, 325:475] = input_data['rgb_focus']
		surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['meta_infos'][1], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['meta_infos'][2], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

		surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
		surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
		surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

		surface[:, 798:802] = 255
		surface[:, 1398:1402] = 255
		surface[:, 1998:2002] = 255


		# display image
		self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))

		pygame.display.flip()
		pygame.event.get()
		return surface

	def _quit(self):
		pygame.quit()



class BasePreprocessor(object):
    """
    Basic Lidar pre-processor.
    Parameters
    ----------
    preprocess_params : dict
        The dictionary containing all parameters of the preprocessing.
    train : bool
        Train or test mode.
    """

    def __init__(self, preprocess_params, train):
        self.params = preprocess_params
        self.train = train


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            from spconv.utils import VoxelGenerator

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.lidar_range,
            max_num_points=self.max_points_per_voxel,
            max_voxels=self.max_voxels
        )

    def preprocess(self, pcd_np):
        data_dict = {}
        voxel_output = self.voxel_generator.generate(pcd_np)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points


        return data_dict

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)

    r1_to_world = np.matrix([[c, -s, t1_x], [s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, -s, t2_x], [s, c, t2_y], [0, 0, 1]])
    # world frame -> r2 frame
    # if r1==r2, do nothing
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


class VLM_Infer():
	def __init__(self, 
              config=None, 
              ego_vehicles_num=1, 
              perception_model=None, 
              planning_model=None, 
              controller=None, 
              perception_dataloader=None, 
              model_config=None, 
              device=None, 
              heter=False, 
              heter_planning_models=None) -> None:
     
		self.config = config
		self._hic = DisplayInterface()
		self.ego_vehicles_num = ego_vehicles_num

		self.memory_measurements = [[], [], [], [], []]
		self.memory_actors_data = [[], [], [], [], []]
		self.det_range = [36, 12, 12, 12, 0.25]
		self.max_distance = 36
		self.distance_to_map_center = (self.det_range[0]+self.det_range[1])/2-self.det_range[1]

		# #### reparse config to retrieve model meta-info
		# model_config = yaml_utils(config["planning"]["planner_config"])

		#### Voxelization Process
		voxel_args = {
			'args': {
				'voxel_size': [0.125, 0.125, 4], # 
				'max_points_per_voxel': 32,
				'max_voxel_train': 70000,
				'max_voxel_test': 40000
			},
			'cav_lidar_range': [-12, -36, -22, 12, 12, 14]   # x_min, y_min, z_min, x_max, y_max, z_max
		}
		# self.voxel_preprocess = SpVoxelPreprocessor(voxel_args, train=False)
	

		self.perception_model = perception_model
		self.planning_model = planning_model
		self.controller = controller
		self.perception_dataloader = perception_dataloader
		self.model_config = model_config
		self.device=device

		self.perception_memory_bank = []
  
		self.heter = heter
		self.heter_planning_models = heter_planning_models
		self.heter_vlm_idxs = self.config['heter']['ego_planner_choice'] if heter else None

		self.input_lidar_size = 224
		self.lidar_range = [36, 36, 36, 36]

		self.softmax = torch.nn.Softmax(dim=0)
		self.traffic_meta_moving_avg = np.zeros((ego_vehicles_num, 400, 7))
		self.prev_lidar = []
		self.prev_control = {}
		self.prev_surround_map = {}

		self.pre_raw_data_bank = {}
		############
		###### multi-agent related components
		############

		### generate the save files for images
		self.skip_frames = self.config['simulation']['skip_frames']
		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
			string += "_".join(
				map(
					lambda x: "%02d" % x,
					(now.month, now.day, now.hour, now.minute, now.second),
				)
			)

			print(string)

			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			(self.save_path / "meta").mkdir(parents=True, exist_ok=False)
   
		# For skipped frames, use the buffer for planning
		self.predicted_result_list_buffer = None
		self.predicted_result_reference_idx = 0
		self.run_time_idx = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


	def get_action_from_list_inter(self, car_data_raw, rsu_data_raw, step, timestamp):
		"""
		Generate the action for N cars from the record data.
		
		Parameters
		----------
		car_data_raw : list of dictionaries and/or None values.
		rsu_data_raw : list of dictionaries and/or None values.
		step : int, the current frame in the simulation (20Hz).
		timestamp : float, current simulation time.
		
		Returns
		-------
		controll_all : list, detailed actions for N cars.
		"""
		# Apply communication latency if configured.
		if 'comm_latency' in self.config['simulation']:
			car_data_raw, rsu_data_raw = self._apply_latency(car_data_raw, rsu_data_raw, step)
		
		# Load data for visualization and planning.
		car_data, car_mask = self.check_data(car_data_raw)
		rsu_data, _ = self.check_data(rsu_data_raw, car=False)
		batch_data = self.collate_batch_infer_perception(car_data, rsu_data)
		
		if self.perception_model is not None:
			# Prepare data for perception.
			extra_source = {'car_data': car_data_raw, 'rsu_data': rsu_data_raw}
			data = self.perception_dataloader.__getitem__(idx=None, extra_source=extra_source)
			batch_data_perception = [data]
			batch_data_perception = self.perception_dataloader.collate_batch_test(batch_data_perception, online_eval_only=True)
			batch_data_perception = train_utils.to_device(batch_data_perception, self.device)
			
			# Process perception if the model is available.
			processed_pred_box_list = self._process_perception(batch_data_perception, car_data_raw, rsu_data_raw, step)
		else:
			processed_pred_box_list = []
		
		
		MEMORY_SIZE = 5
		if step % self.skip_frames == 0 or len(self.perception_memory_bank) == 0 or self.predicted_result_list_buffer is None:
			while len(self.perception_memory_bank) > MEMORY_SIZE:
				self.perception_memory_bank.pop(0)
			self.perception_memory_bank.append({
				'rgb_front': np.stack([car_data_raw[i]['rgb_front'] for i in range(len(car_data_raw))]), # N, H, W, 3
				'rgb_left': np.stack([car_data_raw[i]['rgb_left'] for i in range(len(car_data_raw))]), # N, H, W, 3
				'rgb_right': np.stack([car_data_raw[i]['rgb_right'] for i in range(len(car_data_raw))]), # N, H, W, 3
				'rgb_rear': np.stack([car_data_raw[i]['rgb_rear'] for i in range(len(car_data_raw))]), # N, H, W, 3
				'object_list': [processed_pred_box_list[i] for i in range(len(processed_pred_box_list))],
				'detmap_pose': batch_data['detmap_pose'][:len(car_data_raw)], # N, 3
				"ego_yaw": np.stack([car_data_raw[i]['measurements']['theta'] for i in range(len(car_data_raw))], axis=0), # N, 1
				'target': np.stack([car_data_raw[i]['measurements']['target_point'] for i in range(len(car_data_raw))], axis=0), # N, 2
				'timestamp': timestamp, # float
			})

			if self.heter:
				num_ego, _ = self.perception_memory_bank[-1]['target'].shape
				assert num_ego == self.ego_vehicles_num, f"num of ego in perception memory bank {num_ego} is different from predefined {self.ego_vehicles_num}"
				collab_agent_intent = []
				predicted_result_list = []
				for i, vlm_idx in enumerate(self.heter_vlm_idxs):
					# print(f"call from model {self.config['heter']['avail_heter_planner_configs'][vlm_idx]}, len of heter models {len(self.heter_planning_models)}")
					agent_intent_dict = self.heter_planning_models[vlm_idx].forward_single_intent(self.perception_memory_bank, self.config, i)
					collab_agent_intent.append(agent_intent_dict)
				for i, vlm_idx in enumerate(self.heter_vlm_idxs):
					# print(f"call from model {self.config['heter']['avail_heter_planner_configs'][vlm_idx]}, len of heter models {len(self.heter_planning_models)}")
					pred_result = self.heter_planning_models[vlm_idx].forward_single_collab(self.perception_memory_bank, 
																				self.config, 
																				i, 
																				deepcopy(collab_agent_intent))
					predicted_result_list.append(pred_result)
			else:
				# TODO(XG): current do not support multi-agent image/intent sharing. 
				# But the same functionality can be achieved by heter with the same model.
				raise NotImplementedError("current do not support multi-agent image/intent sharing. But the same functionality can be achieved by heter with the same model.")
				# predicted_result_list = self.planning_model(self.perception_memory_bank, self.config) # [1, 10, 2]
    
			self.perception_memory_bank[-1]['predicted_result_list'] = predicted_result_list
			self.predicted_result_list_buffer = predicted_result_list
			self.predicted_result_reference_idx = 0
		else:
			# """
			# Here, the idea is that if we have the buffer, we will use the buffer to generate the action.
			# Since the buffer contains multiple timestamp, each time we use the buffer, we will pop the first element.
			# If the list length is smaller or equal to 1, we will use the current data to generate the action.
			# [
			# 	# v_idx = 0
			# 	{
			# 		key: [x, x, x, ...]
     		# 		key: [x, x, x, ...]
			# 	}
			# 	# v_idx = 1
			# 	{
			# 		key: [x, x, x, ...]
     		# 		key: [x, x, x, ...]
			# 	}
			# 	...
			# ]
			# """
			# for v_idx in range(len(self.predicted_result_list_buffer)):
			# 	for key in self.predicted_result_list_buffer[v_idx]:
			# 		import pdb; pdb.set_trace()
			# 		if (
         	# 			isinstance(self.predicted_result_list_buffer[v_idx][key], torch.Tensor) or 
			# 			isinstance(self.predicted_result_list_buffer[v_idx][key], np.ndarray) or
			# 			isinstance(self.predicted_result_list_buffer[v_idx][key], list) \
            #  			) \
         	# 			and len(self.predicted_result_list_buffer[v_idx][key]) > 1:
			# 			self.predicted_result_list_buffer[v_idx][key] = self.predicted_result_list_buffer[v_idx][key][1:]
			predicted_result_list = self.predicted_result_list_buffer
			self.predicted_result_reference_idx += 1
   
		# save images for visualization
		if self.heter:
			for i, vlm_idx in enumerate(self.heter_vlm_idxs):
				# Save image for visualization TODO: make the code cleaner
				images = Image.fromarray(car_data_raw[i]['rgb_front'])
				save_dir = pathlib.Path(os.environ['RESULT_ROOT']) / "image_buffer"
				save_dir_run_time = save_dir / self.run_time_idx
				save_dir_agent = save_dir_run_time / f"agent_{i}"
				save_dir_agent.mkdir(parents=True, exist_ok=True)
				image_dir = save_dir_agent / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-buffer.png"
				images.save(image_dir)
		else:
			images = Image.fromarray(car_data_raw[i]['rgb_front'])
			save_dir = pathlib.Path(os.environ['RESULT_ROOT']) / "image_buffer"
			save_dir_run_time = save_dir / self.run_time_idx
			save_dir_agent = save_dir_run_time / f"agent_0"
			save_dir_agent.mkdir(parents=True, exist_ok=True)
			image_dir = save_dir_agent / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-buffer.png"
			images.save(image_dir)

		

		### output postprocess to generate the action, list of actions for N agents
		control_all = self.generate_action_from_model_output(predicted_result_list, car_data_raw, 
                                                       rsu_data_raw, car_data, rsu_data, batch_data, 
                                                       None, car_mask, step, timestamp)
		return control_all

					
				
        
	# def generate_action_from_model_output(self, predicted_result_list, car_data_raw, 
    #                                    rsu_data_raw, car_data, rsu_data, batch_data, planning_input, 
    #                                    car_mask, step, timestamp):
	# 	control_all = []
	# 	tick_data = []
	# 	ego_i = -1
	# 	for count_i in range(self.ego_vehicles_num):
	# 		if not car_mask[count_i]:
	# 			control_all.append(None)
	# 			tick_data.append(None)
	# 			continue

	# 		# store the data for visualization
	# 		tick_data.append({})
	# 		ego_i += 1
	# 		# get the data for current vehicle
	# 		# pred_waypoints = np.around(pred_waypoints_total[ego_i].detach().cpu().numpy(), decimals=2)

	# 		route_info = {
	# 			'speed': car_data_raw[ego_i]['measurements']["speed"],
    # 			'target': car_data_raw[ego_i]['measurements']["target_point"],
	# 			'route_length': 0,
	# 			'route_time': 0,
	# 			'drive_length': 0,
	# 			'drive_time': 0
	# 		}
   
	# 		route_info.update(predicted_result_list[ego_i])
			
	# 		print(f"router information: {route_info}")

	# 		steer, throttle, brake, meta_infos = self.controller[ego_i].run_step(
	# 			route_info
	# 		)

	# 		control = carla.VehicleControl()
	# 		control.steer = float(steer)
	# 		control.throttle = float(throttle)
	# 		control.brake = float(brake)

	# 		self.prev_control[ego_i] = control


	# 		control_all.append(control)

	# 		# 添加 BEV 相机数据（如果存在）并进行可视化
	# 		if 'rgb_bev' in car_data_raw[ego_i] and car_data_raw[ego_i]['rgb_bev'] is not None:
	# 			# 获取原始 BEV 图像
	# 			bev_img = car_data_raw[ego_i]["rgb_bev"].copy()
	# 			H, W = bev_img.shape[:2]
	# 			image_center = np.array([W//2, H//2])
	# 			pixels_per_meter = 20  # 每米对应的像素数

	# 			# 在图像上添加速度信息
	# 			speed_text = f"Speed: {route_info['speed']:.2f} m/s"
	# 			cv2.putText(bev_img, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

	# 			# 绘制车辆起始点（图像中心）
	# 			# 绘制一个蓝色圆圈表示车辆位置
	# 			# cv2.circle(bev_img, (image_center[0], image_center[1]), 6, (255, 0, 0), 2)
	# 			# 绘制一个小箭头表示车辆朝向（向上）
	# 			arrow_length = 15
	# 			cv2.arrowedLine(bev_img, 
	# 				(image_center[0], image_center[1]), 
	# 				(image_center[0], image_center[1] - arrow_length), 
	# 				(255, 0, 0), 2)

	# 			# 根据不同类型的预测结果进行可视化
	# 			# 1. Waypoints 类型
	# 			if 'waypoints' in route_info:
	# 				waypoints = route_info['waypoints']
	# 				# 打印调试信息
	# 				print(f"Waypoints: {waypoints}")
					
	# 				# 查看一下第一个点的坐标
	# 				if len(waypoints) > 0:
	# 					print(f"First waypoint: x={waypoints[0][0]}, y={waypoints[0][1]}")
					
	# 				for i in range(len(waypoints)):
	# 					# 在 BEV 图像中：
	# 					# - 车辆前方是 -y方向
	# 					# - 车辆右侧是 +x方向
	# 					# 根据图片分析，waypoints的坐标系与我们的假设不同
						
	# 					# 假设 waypoints[i][0] 是横向偏移（x轴）
	# 					# 假设 waypoints[i][1] 是前向距离（y轴）
	# 					pt_x = int(image_center[0] + waypoints[i][0] * pixels_per_meter)
	# 					pt_y = int(image_center[1] + waypoints[i][1] * pixels_per_meter)
						
	# 					if 0 <= pt_x < W and 0 <= pt_y < H:
	# 						# 绘制点，不连线
	# 						cv2.circle(bev_img, (pt_x, pt_y), 4, (0, 255, 0), -1)
	# 						# 添加点的序号
	# 						# cv2.putText(bev_img, str(i), (pt_x+5, pt_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

				
	# 			elif 'target_speed' in route_info and 'curvature' in route_info:
	# 				# 显示目标速度和曲率信息
	# 				target_speed = route_info['target_speed'][0]
	# 				curvature = route_info['curvature'][0] / 10
	# 				speed_text = f"Target Speed: {target_speed:.2f} m/s"
	# 				cv2.putText(bev_img, speed_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
	# 				curv_text = f"Curvature: {curvature:.3f} degree/m"
	# 				cv2.putText(bev_img, curv_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
					
	# 				# 获取预测点的数量
	# 				num_points = min(len(route_info['target_speed']), len(route_info['curvature']))
	# 				dt = route_info['dt']
					
	# 				# 起始位置和方向
	# 				current_x = 0
	# 				current_y = 0
	# 				current_yaw = 0
					
	# 				# 存储轨迹点（图像坐标）
	# 				image_points = []
	# 				image_points.append((int(image_center[0]), int(image_center[1])))
					
	# 				# 处理每个预测点
	# 				for i in range(num_points):
	# 					speed = route_info['target_speed'][i]
	# 					curv = np.deg2rad(route_info['curvature'][i]/10)
						
	# 					# 使用多个子步骤创建平滑曲线
	# 					num_substeps = 10
	# 					substep_dt = dt / num_substeps
						
	# 					for _ in range(num_substeps):
	# 						# 计算在这个子步骤中行驶的距离
	# 						substep_dist = speed * substep_dt
							
	# 						# 计算中点偏航角以提高精度
	# 						mid_yaw = current_yaw + (curv * substep_dist) / 2
							
	# 						# 使用中点偏航角更新位置
	# 						current_x += substep_dist * np.cos(mid_yaw)
	# 						current_y += substep_dist * np.sin(mid_yaw)
							
	# 						# 更新完整子步骤的偏航角
	# 						current_yaw += curv * substep_dist
							
	# 						# 转换为图像坐标并添加到轨迹
	# 						img_y = int(image_center[0] - current_x * pixels_per_meter)
	# 						img_x = int(image_center[1] + current_y * pixels_per_meter)
							
	# 						if 0 <= img_x < W and 0 <= img_y < H:
	# 							image_points.append((img_x, img_y))
					
	# 				# 轨迹宽度（像素）
	# 				traj_width_pixels = 80
					
	# 				# 创建左右两侧的点（图像坐标系）
	# 				left_side = []
	# 				right_side = []
					
	# 				for i in range(len(image_points)):
	# 					# 计算当前点的方向向量
	# 					if i == 0 and len(image_points) > 1:
	# 						# 第一个点，使用下一个点的方向
	# 						dx = image_points[1][0] - image_points[0][0]
	# 						dy = image_points[1][1] - image_points[0][1]
	# 					elif i == len(image_points) - 1 and i > 0:
	# 						# 最后一个点，使用前一个点的方向
	# 						dx = image_points[i][0] - image_points[i-1][0]
	# 						dy = image_points[i][1] - image_points[i-1][1]
	# 					elif 0 < i < len(image_points) - 1:
	# 						# 中间点，使用前后点的平均方向
	# 						dx1 = image_points[i][0] - image_points[i-1][0]
	# 						dy1 = image_points[i][1] - image_points[i-1][1]
	# 						dx2 = image_points[i+1][0] - image_points[i][0]
	# 						dy2 = image_points[i+1][1] - image_points[i][1]
	# 						dx = (dx1 + dx2) / 2
	# 						dy = (dy1 + dy2) / 2
	# 					else:
	# 						# 只有一个点，无法确定方向
	# 						continue
						
	# 					# 标准化方向向量
	# 					norm = np.sqrt(dx*dx + dy*dy)
	# 					if norm < 1e-6:  # 避免除以零
	# 						continue
	# 					dx, dy = dx/norm, dy/norm
						
	# 					# 计算法向量（垂直于方向向量）
	# 					nx, ny = -dy, dx  # 逆时针旋转90度
						
	# 					# 计算左右两侧的点（在图像坐标系中）
	# 					half_width = traj_width_pixels / 2
	# 					left_x = int(image_points[i][0] + nx * half_width)
	# 					left_y = int(image_points[i][1] + ny * half_width)
	# 					right_x = int(image_points[i][0] - nx * half_width)
	# 					right_y = int(image_points[i][1] - ny * half_width)
						
	# 					if (0 <= left_x < W and 0 <= left_y < H and 
	# 						0 <= right_x < W and 0 <= right_y < H):
	# 						left_side.append((left_x, left_y))
	# 						right_side.append((right_x, right_y))
					
	# 				# 创建轨迹多边形
	# 				if len(left_side) > 0 and len(right_side) > 0:
	# 					# 创建一个空白的图层
	# 					overlay = np.zeros_like(bev_img)
						
	# 					# 将左右两侧的点组合成一个多边形
	# 					polygon = np.array(left_side + list(reversed(right_side)), dtype=np.int32)
						
	# 					# 填充多边形
	# 					cv2.fillPoly(overlay, [polygon], (152, 214, 152))
						
	# 					# 创建掩码
	# 					mask = np.any(overlay != 0, axis=2)
						
	# 					# 应用透明度
	# 					alpha = 0.5
	# 					bev_img[mask] = cv2.addWeighted(bev_img, alpha, overlay, 1 - alpha, 0)[mask]


	# 			# 3. Control 类型
	# 			elif 'steering' in route_info and 'throttle' in route_info and 'brake' in route_info:
	# 				# 显示控制信号
	# 				steering = route_info['steering'][0]  # 使用第一个预测值
	# 				throttle = route_info['throttle'][0]
	# 				brake = route_info['brake'][0]
	# 				control_text = f"Steering: {steering:.2f}, Throttle: {throttle:.2f}, Brake: {brake:.2f}"
	# 				cv2.putText(bev_img, control_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

	# 				# 绘制转向预测方向
	# 				arrow_length = 50
	# 				arrow_angle = steering * np.pi / 4  # 将转向值映射到角度
	# 				end_x = int(image_center[0] + arrow_length * np.sin(arrow_angle))
	# 				end_y = int(image_center[1] + arrow_length * np.cos(arrow_angle))
	# 				cv2.arrowedLine(bev_img, (image_center[0], image_center[1]), (end_x, end_y), (255, 0, 0), 2)

	# 			# 绘制目标点
	# 			# 由于target point和waypoints的xy值相反，需要先取负再转换
	# 			target_x = int(image_center[0] + (route_info['target'][0]) * pixels_per_meter)  # 横向偏移
	# 			target_y = int(image_center[1] - (-route_info['target'][1]) * pixels_per_meter)  # 前向距离
	# 			if 0 <= target_x < W and 0 <= target_y < H:
	# 				# 画线从当前位置到目标点
	# 				# cv2.line(bev_img, (image_center[0], image_center[1]), (target_x, target_y), (0, 0, 255), 2)
	# 				# 在目标点画一个红色实心圆
	# 				cv2.circle(bev_img, (target_x, target_y), 15, (255, 0, 0), -1)
	# 				# 添加标签
	# 				cv2.putText(bev_img, "Target", (target_x-60, target_y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

	# 			tick_data[ego_i]["rgb_bev"] = bev_img
	# 			# 将 BEV 相机图像调整为统一大小供显示
	# 			# tick_data[ego_i]["rgb_bev_display"] = cv2.resize(bev_img, (300, 300))
	# 			tick_data[ego_i]["rgb_bev_display"] = bev_img
			
		
	# 	if SAVE_PATH is not None:
	# 		self.save(tick_data, step)
		
	# 	return control_all


	def generate_action_from_model_output(self, predicted_result_list, car_data_raw, 
										rsu_data_raw, car_data, rsu_data, batch_data, planning_input, 
										car_mask, step, timestamp):
		control_all = []
		tick_data = []
		ego_i = -1
		for count_i in range(self.ego_vehicles_num):
			if not car_mask[count_i]:
				control_all.append(None)
				tick_data.append(None)
				continue

			# store the data for visualization
			tick_data.append({})
			ego_i += 1
			# get the data for current vehicle

			route_info = {
				'speed': car_data_raw[ego_i]['measurements']["speed"],
				'target': car_data_raw[ego_i]['measurements']["target_point"],
				'route_length': 0,
				'route_time': 0,
				'drive_length': 0,
				'drive_time': 0
			}
			
			route_info.update(predicted_result_list[ego_i])
			
			print(f"router information: {route_info}")

			steer, throttle, brake, meta_infos = self.controller[ego_i].run_step(
				route_info, buffer_idx=self.predicted_result_reference_idx
			)

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)

			self.prev_control[ego_i] = control

			control_all.append(control)

			# Add BEV camera data (if exists) and visualize
			if 'rgb_bev' in car_data_raw[ego_i] and car_data_raw[ego_i]['rgb_bev'] is not None:
				# Get original BEV image
				bev_img = car_data_raw[ego_i]["rgb_bev"].copy()
				H, W = bev_img.shape[:2]
				image_center = np.array([W//2, H//2])
				pixels_per_meter = 40  # Pixels per meter 1600 / 40 = 40 TODO: make it configurable

				# Draw common elements (speed info, vehicle position, target point)
				bev_img = self._draw_common_elements(bev_img, route_info, image_center, pixels_per_meter, W, H)

				# Draw visualization based on prediction type
				if 'waypoints' in route_info:
					bev_img = self._draw_waypoints_based_trajectory(bev_img, route_info, image_center, pixels_per_meter, W, H)
				elif 'target_speed' in route_info and 'curvature' in route_info:
					bev_img = self._draw_speed_curvature_based_trajectory(bev_img, route_info, image_center, pixels_per_meter, W, H)
				elif 'steering' in route_info and 'throttle' in route_info and 'brake' in route_info:
					bev_img = self._draw_control_based_trajectory(bev_img, route_info, image_center, pixels_per_meter, W, H)

				tick_data[ego_i]["rgb_bev"] = bev_img
				tick_data[ego_i]["rgb_bev_display"] = bev_img
			
		if SAVE_PATH is not None:
			self.save(tick_data, step)
		
		return control_all

	def _draw_common_elements(self, bev_img, route_info, image_center, pixels_per_meter, W, H):
		"""
		Draw common elements like speed information, vehicle position and target point
		
		Args:
			bev_img: BEV image
			route_info: Route information dictionary
			image_center: Center position of the image
			pixels_per_meter: Pixels per meter ratio
			W: Image width
			H: Image height
			
		Returns:
			bev_img: Updated BEV image
		"""
		# Display speed information
		speed_text = f"Speed: {route_info['speed']:.2f} m/s"
		cv2.putText(bev_img, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
		
		# Draw a small arrow indicating the vehicle heading (upward)
		arrow_length = 15
		cv2.arrowedLine(bev_img, 
			(image_center[0], image_center[1]), 
			(image_center[0], image_center[1] - arrow_length), 
			(255, 0, 0), 2)
		
		# Draw target point
		target_x = int(image_center[0] + (route_info['target'][0]) * pixels_per_meter)
		target_y = int(image_center[1] - (-route_info['target'][1]) * pixels_per_meter)
		if 0 <= target_x < W and 0 <= target_y < H:
			cv2.circle(bev_img, (target_x, target_y), 15, (255, 0, 0), -1)
			cv2.putText(bev_img, "Target", (target_x-60, target_y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
		
		return bev_img

	def _draw_waypoints_based_trajectory(self, bev_img, route_info, image_center, pixels_per_meter, W, H):
		"""
		Draw waypoints-based trajectory with width and transparency
		
		Args:
			bev_img: BEV image
			route_info: Route information dictionary
			image_center: Center position of the image
			pixels_per_meter: Pixels per meter ratio
			W: Image width
			H: Image height
			
		Returns:
			bev_img: Updated BEV image
		"""
		waypoints = route_info['waypoints']
		print(f"Waypoints: {waypoints}")
		
		if len(waypoints) > 0:
			print(f"First waypoint: x={waypoints[0][0]}, y={waypoints[0][1]}")
		
		# Convert waypoints to image coordinates
		image_points = []
		image_points.append((int(image_center[0]), int(image_center[1])))  # Start from vehicle position
		
		for i in range(len(waypoints)):
			pt_x = int(image_center[0] + waypoints[i][0] * pixels_per_meter)
			pt_y = int(image_center[1] + waypoints[i][1] * pixels_per_meter)
			
			if 0 <= pt_x < W and 0 <= pt_y < H:
				image_points.append((pt_x, pt_y))
		
		# Trajectory width in pixels
		traj_width_pixels = 80
		
		# Create left and right side points (in image coordinate system)
		left_side = []
		right_side = []
		
		for i in range(len(image_points)):
			# Calculate direction vector for current point
			if i == 0 and len(image_points) > 1:
				# First point, use direction to next point
				dx = image_points[1][0] - image_points[0][0]
				dy = image_points[1][1] - image_points[0][1]
			elif i == len(image_points) - 1 and i > 0:
				# Last point, use direction from previous point
				dx = image_points[i][0] - image_points[i-1][0]
				dy = image_points[i][1] - image_points[i-1][1]
			elif 0 < i < len(image_points) - 1:
				# Middle point, use average direction
				dx1 = image_points[i][0] - image_points[i-1][0]
				dy1 = image_points[i][1] - image_points[i-1][1]
				dx2 = image_points[i+1][0] - image_points[i][0]
				dy2 = image_points[i+1][1] - image_points[i][1]
				dx = (dx1 + dx2) / 2
				dy = (dy1 + dy2) / 2
			else:
				# Only one point, can't determine direction
				continue
			
			# Normalize direction vector
			norm = np.sqrt(dx*dx + dy*dy)
			if norm < 1e-6:  # Avoid division by zero
				continue
			dx, dy = dx/norm, dy/norm
			
			# Calculate normal vector (perpendicular to direction vector)
			nx, ny = -dy, dx  # Rotate 90 degrees counter-clockwise
			
			# Calculate left and right side points (in image coordinate system)
			half_width = traj_width_pixels / 2
			left_x = int(image_points[i][0] + nx * half_width)
			left_y = int(image_points[i][1] + ny * half_width)
			right_x = int(image_points[i][0] - nx * half_width)
			right_y = int(image_points[i][1] - ny * half_width)
			
			if (0 <= left_x < W and 0 <= left_y < H and 
				0 <= right_x < W and 0 <= right_y < H):
				left_side.append((left_x, left_y))
				right_side.append((right_x, right_y))
		
		# Create trajectory polygon
		if len(left_side) > 0 and len(right_side) > 0:
			# Create an empty overlay
			overlay = np.zeros_like(bev_img)
			
			# Combine left and right side points into a polygon
			polygon = np.array(left_side + list(reversed(right_side)), dtype=np.int32)
			
			# Fill polygon
			cv2.fillPoly(overlay, [polygon], (152, 214, 152))  # Light green color
			
			# Create mask
			mask = np.any(overlay != 0, axis=2)
			
			# Apply transparency
			alpha = 0.5
			bev_img[mask] = cv2.addWeighted(bev_img, alpha, overlay, 1 - alpha, 0)[mask]
		
		# Additionally draw each waypoint as a circle
		for i in range(len(waypoints)):
			pt_x = int(image_center[0] + waypoints[i][0] * pixels_per_meter)
			pt_y = int(image_center[1] + waypoints[i][1] * pixels_per_meter)
			
			if 0 <= pt_x < W and 0 <= pt_y < H:
				cv2.circle(bev_img, (pt_x, pt_y), 4, (0, 255, 0), -1)
		
		return bev_img

	def _draw_speed_curvature_based_trajectory(self, bev_img, route_info, image_center, pixels_per_meter, W, H):
		"""
		Draw speed-curvature based trajectory with width and transparency
		
		Args:
			bev_img: BEV image
			route_info: Route information dictionary
			image_center: Center position of the image
			pixels_per_meter: Pixels per meter ratio
			W: Image width
			H: Image height
			
		Returns:
			bev_img: Updated BEV image
		"""
		# Display target speed and curvature information
		target_speed = route_info['target_speed'][0]
		curvature = route_info['curvature'][0] / 10 # Here, 10 is a hardcoded value to scale down curvature slightly
		speed_text = f"Target Speed: {target_speed:.2f} m/s"
		cv2.putText(bev_img, speed_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
		curv_text = f"Curvature: {curvature:.3f} degree/m"
		cv2.putText(bev_img, curv_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
		
		# Get number of prediction points
		num_points = min(len(route_info['target_speed']), len(route_info['curvature']))
		dt = route_info['dt']
		
		# Initial position and direction
		current_x = 0
		current_y = 0
		current_yaw = 0
		
		# Store trajectory points (image coordinates)
		image_points = []
		image_points.append((int(image_center[0]), int(image_center[1])))
		
		# Process each prediction point
		for i in range(num_points):
			speed = route_info['target_speed'][i]
			curv = np.deg2rad(route_info['curvature'][i]/10)
			
			# Use multiple substeps to create a smooth curve
			num_substeps = 10
			substep_dt = dt / num_substeps
			
			for _ in range(num_substeps):
				# Calculate distance traveled in this substep
				substep_dist = speed * substep_dt
				
				# Calculate midpoint yaw for improved accuracy
				mid_yaw = current_yaw + (curv * substep_dist) / 2
				
				# Update position using midpoint yaw
				current_x += substep_dist * np.cos(mid_yaw)
				current_y += substep_dist * np.sin(mid_yaw)
				
				# Update full substep yaw
				current_yaw += curv * substep_dist
				
				# Convert to image coordinates and add to trajectory
				img_y = int(image_center[0] - current_x * pixels_per_meter)
				img_x = int(image_center[1] + current_y * pixels_per_meter)
				
				if 0 <= img_x < W and 0 <= img_y < H:
					image_points.append((img_x, img_y))
		
		# Trajectory width in pixels
		traj_width_pixels = 80
		
		# Create left and right side points (in image coordinate system)
		left_side = []
		right_side = []
		
		for i in range(len(image_points)):
			# Calculate direction vector for current point
			if i == 0 and len(image_points) > 1:
				# First point, use direction to next point
				dx = image_points[1][0] - image_points[0][0]
				dy = image_points[1][1] - image_points[0][1]
			elif i == len(image_points) - 1 and i > 0:
				# Last point, use direction from previous point
				dx = image_points[i][0] - image_points[i-1][0]
				dy = image_points[i][1] - image_points[i-1][1]
			elif 0 < i < len(image_points) - 1:
				# Middle point, use average direction
				dx1 = image_points[i][0] - image_points[i-1][0]
				dy1 = image_points[i][1] - image_points[i-1][1]
				dx2 = image_points[i+1][0] - image_points[i][0]
				dy2 = image_points[i+1][1] - image_points[i][1]
				dx = (dx1 + dx2) / 2
				dy = (dy1 + dy2) / 2
			else:
				# Only one point, can't determine direction
				continue
			
			# Normalize direction vector
			norm = np.sqrt(dx*dx + dy*dy)
			if norm < 1e-6:  # Avoid division by zero
				continue
			dx, dy = dx/norm, dy/norm
			
			# Calculate normal vector (perpendicular to direction vector)
			nx, ny = -dy, dx  # Rotate 90 degrees counter-clockwise
			
			# Calculate left and right side points (in image coordinate system)
			half_width = traj_width_pixels / 2
			left_x = int(image_points[i][0] + nx * half_width)
			left_y = int(image_points[i][1] + ny * half_width)
			right_x = int(image_points[i][0] - nx * half_width)
			right_y = int(image_points[i][1] - ny * half_width)
			
			if (0 <= left_x < W and 0 <= left_y < H and 
				0 <= right_x < W and 0 <= right_y < H):
				left_side.append((left_x, left_y))
				right_side.append((right_x, right_y))
		
		# Create trajectory polygon
		if len(left_side) > 0 and len(right_side) > 0:
			# Create an empty overlay
			overlay = np.zeros_like(bev_img)
			
			# Combine left and right side points into a polygon
			polygon = np.array(left_side + list(reversed(right_side)), dtype=np.int32)
			
			# Fill polygon
			cv2.fillPoly(overlay, [polygon], (152, 214, 152))
			
			# Create mask
			mask = np.any(overlay != 0, axis=2)
			
			# Apply transparency
			alpha = 0.5
			bev_img[mask] = cv2.addWeighted(bev_img, alpha, overlay, 1 - alpha, 0)[mask]
		
		return bev_img

	def _draw_control_based_trajectory(self, bev_img, route_info, image_center, pixels_per_meter, W, H):
		"""
		Draw control-based trajectory with width and transparency
		
		Args:
			bev_img: BEV image
			route_info: Route information dictionary
			image_center: Center position of the image
			pixels_per_meter: Pixels per meter ratio
			W: Image width
			H: Image height
			
		Returns:
			bev_img: Updated BEV image
		"""
		# Display control signals
		steering_angles = route_info['steering']
		throttle_values = route_info['throttle']
		brake_values = route_info['brake']
		
		steering = steering_angles[0]  # Use first prediction value
		throttle = throttle_values[0]
		brake = brake_values[0]
		
		control_text = f"Steering: {steering:.2f}, Throttle: {throttle:.2f}, Brake: {brake:.2f}"
		cv2.putText(bev_img, control_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
		
		# Create trajectory points based on steering predictions
		dt = 0.1  # Time step (assuming 0.1s if not provided)
		if 'dt' in route_info:
			dt = route_info['dt']
		
		# Initial position and direction
		current_x = 0
		current_y = 0
		current_yaw = 0
		
		# Estimate speed from throttle/brake (simple model)
		speed = route_info['speed']  # Start with current speed
		
		# Store trajectory points (image coordinates)
		image_points = []
		image_points.append((int(image_center[0]), int(image_center[1])))
		
		# Process each control point
		num_points = min(len(steering_angles), len(throttle_values), len(brake_values))
		
		for i in range(num_points):
			steering = steering_angles[i]
			throttle = throttle_values[i]
			brake = brake_values[i]
			
			# Simple vehicle dynamics model
			# Update speed based on throttle/brake (very simplified)
			accel = (throttle - brake) * 3.0  # Simple acceleration model
			speed += accel * dt
			speed = max(0, min(speed, 30))  # Limit speed
			
			# Steering angle to curvature conversion (simplified)
			curvature = steering * 0.5  # Map steering [-1,1] to curvature
			
			# Use multiple substeps to create a smooth curve
			num_substeps = 10
			substep_dt = dt / num_substeps
			
			for _ in range(num_substeps):
				# Calculate distance traveled in this substep
				substep_dist = speed * substep_dt
				
				# Calculate midpoint yaw for improved accuracy
				mid_yaw = current_yaw + (curvature * substep_dist) / 2
				
				# Update position using midpoint yaw
				current_x += substep_dist * np.cos(mid_yaw)
				current_y += substep_dist * np.sin(mid_yaw)
				
				# Update full substep yaw
				current_yaw += curvature * substep_dist
				
				# Convert to image coordinates and add to trajectory
				img_y = int(image_center[0] - current_x * pixels_per_meter)
				img_x = int(image_center[1] + current_y * pixels_per_meter)
				
				if 0 <= img_x < W and 0 <= img_y < H:
					image_points.append((img_x, img_y))
		
		# Trajectory width in pixels
		traj_width_pixels = 80
		
		# Create left and right side points (in image coordinate system)
		left_side = []
		right_side = []
		
		for i in range(len(image_points)):
			# Calculate direction vector for current point
			if i == 0 and len(image_points) > 1:
				# First point, use direction to next point
				dx = image_points[1][0] - image_points[0][0]
				dy = image_points[1][1] - image_points[0][1]
			elif i == len(image_points) - 1 and i > 0:
				# Last point, use direction from previous point
				dx = image_points[i][0] - image_points[i-1][0]
				dy = image_points[i][1] - image_points[i-1][1]
			elif 0 < i < len(image_points) - 1:
				# Middle point, use average direction
				dx1 = image_points[i][0] - image_points[i-1][0]
				dy1 = image_points[i][1] - image_points[i-1][1]
				dx2 = image_points[i+1][0] - image_points[i][0]
				dy2 = image_points[i+1][1] - image_points[i][1]
				dx = (dx1 + dx2) / 2
				dy = (dy1 + dy2) / 2
			else:
				# Only one point, can't determine direction
				continue
			
			# Normalize direction vector
			norm = np.sqrt(dx*dx + dy*dy)
			if norm < 1e-6:  # Avoid division by zero
				continue
			dx, dy = dx/norm, dy/norm
			
			# Calculate normal vector (perpendicular to direction vector)
			nx, ny = -dy, dx  # Rotate 90 degrees counter-clockwise
			
			# Calculate left and right side points (in image coordinate system)
			half_width = traj_width_pixels / 2
			left_x = int(image_points[i][0] + nx * half_width)
			left_y = int(image_points[i][1] + ny * half_width)
			right_x = int(image_points[i][0] - nx * half_width)
			right_y = int(image_points[i][1] - ny * half_width)
			
			if (0 <= left_x < W and 0 <= left_y < H and 
				0 <= right_x < W and 0 <= right_y < H):
				left_side.append((left_x, left_y))
				right_side.append((right_x, right_y))
		
		# Create trajectory polygon
		if len(left_side) > 0 and len(right_side) > 0:
			# Create an empty overlay
			overlay = np.zeros_like(bev_img)
			
			# Combine left and right side points into a polygon
			polygon = np.array(left_side + list(reversed(right_side)), dtype=np.int32)
			
			# Fill polygon
			cv2.fillPoly(overlay, [polygon], (152, 214, 152))
			
			# Create mask
			mask = np.any(overlay != 0, axis=2)
			
			# Apply transparency
			alpha = 0.5
			bev_img[mask] = cv2.addWeighted(bev_img, alpha, overlay, 1 - alpha, 0)[mask]
		
		# Draw simple steering direction arrow
		arrow_length = 50
		arrow_angle = steering * np.pi / 4  # Map steering value to angle
		end_x = int(image_center[0] + arrow_length * np.sin(arrow_angle))
		end_y = int(image_center[1] - arrow_length * np.cos(arrow_angle))
		cv2.arrowedLine(bev_img, (image_center[0], image_center[1]), (end_x, end_y), (255, 0, 0), 2)
		
		return bev_img


	def _apply_latency(self, car_data_raw, rsu_data_raw, step):
		"""
		Handle communication latency by retrieving data from a previous time step.
		
		Parameters:
		- car_data_raw: list of raw car data.
		- rsu_data_raw: list of raw RSU data.
		- step: current time step.
		
		Returns:
		- Updated car_data_raw and rsu_data_raw after applying latency.
		"""
		raw_data_dict = {'car_data': car_data_raw, 'rsu_data': rsu_data_raw}
		self.pre_raw_data_bank.update({step: raw_data_dict})
		latency_step = self.config['simulation']['comm_latency']
		sorted_keys = sorted(list(self.pre_raw_data_bank.keys()))
		if step > latency_step:
			if step - sorted_keys[0] > latency_step:
				self.pre_raw_data_bank.pop(sorted_keys[0])
			if step - latency_step in self.pre_raw_data_bank:
				raw_data_used = self.pre_raw_data_bank[step - latency_step]
				for i in range(len(car_data_raw)):
					if i > 0:
						car_data_raw[i] = raw_data_used['car_data'][i]
				rsu_data_raw = raw_data_used['rsu_data']
			else:
				print('Latency data not found!')
		return car_data_raw, rsu_data_raw

	def _process_perception(self, batch_data_perception, car_data_raw, rsu_data_raw, step):
		"""
		Process the perception model output including post-processing of prediction boxes.
		
		Parameters:
		- batch_data_perception: collated batch data for perception.
		- car_data_raw: raw car data (used to obtain measurements).
		- rsu_data_raw: raw RSU data.
		- step: current time step.
		
		Returns:
		- processed_pred_box_list: list of processed prediction boxes for each vehicle.
		- infer_result: dictionary with additional inference results.
		"""
		# Forward pass through perception model for each agent.
		output_dict = OrderedDict()
		for cav_id, cav_content in batch_data_perception.items():
			output_dict[cav_id] = self.perception_model(cav_content)
			
		# Post-process perception results.
		pred_box_tensor, pred_score, gt_box_tensor = self.perception_dataloader.post_process_multiclass_no_fusion(
			batch_data_perception, output_dict, online_eval_only=True)
		infer_result = {"pred_box_tensor": pred_box_tensor,
						"pred_score": pred_score,
						"gt_box_tensor": gt_box_tensor}
		if "comm_rate" in output_dict.get('ego', {}):
			infer_result.update({"comm_rate": output_dict['ego']['comm_rate']})

		# Process each agent's predictions.
		processed_pred_box_list = []
		for cav_id in range(len(pred_box_tensor)):
			attrib_list = ['pred_box_tensor', 'pred_score', 'gt_box_tensor']
			for attrib in attrib_list:
				if isinstance(infer_result[attrib][cav_id], list):
					infer_result_tensor = [item for item in infer_result[attrib][cav_id] if item is not None]
					infer_result[attrib][cav_id] = torch.cat(infer_result_tensor, dim=0) if infer_result_tensor else None

			folder_path = self.save_path / pathlib.Path("ego_vehicle_{}".format(0))
			if not os.path.exists(folder_path):
				os.mkdir(folder_path)
				
			# Filter out ego box and process prediction boxes.
			if infer_result['pred_box_tensor'][cav_id] is not None:
				if len(infer_result['pred_box_tensor'][cav_id]) > 0:
					tmp = infer_result['pred_box_tensor'][cav_id][:, :, 0].clone()
					infer_result['pred_box_tensor'][cav_id][:, :, 0] = infer_result['pred_box_tensor'][cav_id][:, :, 1]
					infer_result['pred_box_tensor'][cav_id][:, :, 1] = tmp
				measurements = car_data_raw[0]['measurements']
				num_object = infer_result['pred_box_tensor'][cav_id].shape[0]
				object_list = []
				for i in range(num_object):
					transformed_box = transform_2d_points(
						infer_result['pred_box_tensor'][cav_id][i].cpu().numpy(),
						np.pi/2 - measurements["theta"],
						measurements["lidar_pose_y"],
						measurements["lidar_pose_x"],
						np.pi/2 - measurements["theta"],
						measurements["y"],
						measurements["x"],
					)
					location_box = np.mean(transformed_box[:4, :2], 0)
					if np.linalg.norm(location_box) < 1.4:
						continue
					object_list.append(torch.from_numpy(transformed_box))
				processed_pred_box = torch.stack(object_list, dim=0) if object_list else infer_result['pred_box_tensor'][cav_id][:0]
			else:
				processed_pred_box = []
			processed_pred_box_list.append(processed_pred_box)
		
		return processed_pred_box_list


	def save(self, tick_data, frame):
		if frame % self.skip_frames != 0:
			return
		for ego_i in range(self.ego_vehicles_num):
			folder_path = self.save_path / pathlib.Path("ego_vehicle_{}".format(ego_i))
			if not os.path.exists(folder_path):
				os.mkdir(folder_path)
			
			# Create a directory for storing BEV camera images
			bev_frames_path = folder_path / "bev_frames"
			if not os.path.exists(bev_frames_path):
				os.mkdir(bev_frames_path)
			
			# Save the main screen image
			# Image.fromarray(tick_data[ego_i]["surface"]).save(
			# 	folder_path / ("%04d.jpg" % frame)
			# )
			
			# Save the BEV map image (generated using BirdViewProducer)
			if "bev" in tick_data[ego_i]:
				map_data = np.array(tick_data[ego_i]["bev"])
				# Normalize data range
				map_data = np.uint8(map_data)
				Image.fromarray(map_data).save(
					bev_frames_path / ("map_%04d.jpg" % frame)
				)
			
			# Save the actual image captured by the BEV camera
			if "rgb_bev" in tick_data[ego_i] and tick_data[ego_i]["rgb_bev"] is not None:
				rgb_bev_data = tick_data[ego_i]["rgb_bev"]
				# Normalize data range
				rgb_bev_data = np.uint8(rgb_bev_data)
				Image.fromarray(rgb_bev_data).save(
					bev_frames_path / ("camera_bev_%04d.jpg" % frame)
				)
			
			# # Save path planning data
			# with open(folder_path / ("%04d.json" % frame), 'w') as f:
			# 	json.dump(tick_data[ego_i]['planning'], f, indent=4)
		return


	

	def check_data(self, raw_data, car=True):
		mask = []
		data = [] # without None
		for i in raw_data:
			if i is not None:
				mask.append(1) # filter the data!
				data.append(self.preprocess_data(copy.deepcopy(i), car=car))
			else:
				mask.append(0)
				data.append(0)
		return data, mask
	

	def preprocess_data(self, data, car=True):
		output_record = {
		}
		
		##########
		## load and pre-process images
		##########
		
		##########
		## load environment data and control signal
		##########    
		measurements = data['measurements']
		cmd_one_hot = [0, 0, 0, 0, 0, 0]
		if not car:
			measurements['command'] = -1
			measurements["speed"] = 0
			measurements['target_point'] = np.array([0, 0])
		cmd = measurements['command'] - 1
		if cmd < 0:
			cmd = 3
		cmd_one_hot[cmd] = 1
		cmd_one_hot.append(measurements["speed"])
		mes = np.array(cmd_one_hot)
		mes = torch.from_numpy(mes).cuda().float()

		output_record["measurements"] = mes
		output_record['command'] = cmd

		lidar_pose_x = measurements["lidar_pose_x"]
		lidar_pose_y = measurements["lidar_pose_y"]
		lidar_theta = measurements["theta"] + np.pi
		
		output_record['lidar_pose'] = np.array([-lidar_pose_y, lidar_pose_x, lidar_theta])

		## 计算density map中心点的世界坐标，目前density map预测范围为左右10m前18m后2m
		detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
		detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
		detmap_theta = measurements["theta"] + np.pi/2
		output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
		output_record["target_point"] = torch.from_numpy(measurements['target_point']).cuda().float()
		
		##########
		## load and pre-process LiDAR from 3D point cloud to 2D map
		##########
		lidar_unprocessed = data['lidar'][:, :3]
		# print(lidar_unprocessed.shape)
		lidar_unprocessed[:, 1] *= -1
		if not car:
			lidar_unprocessed[:, 2] = lidar_unprocessed[:, 2] + np.array([measurements["lidar_pose_z"]])[np.newaxis, :] - np.array([2.1])[np.newaxis, :] 
		
		lidar_processed = self.lidar_to_histogram_features(
			lidar_unprocessed, crop=self.input_lidar_size, lidar_range=self.lidar_range
		)        
		# if self.lidar_transform is not None:
		# 	lidar_processed = self.lidar_transform(lidar_processed)
		output_record["lidar_original"] = lidar_processed

		# lidar_unprocessed[:, 0] *= -1

		# voxel_dict = self.voxel_preprocess.preprocess(lidar_unprocessed)
		# output_record["lidar"] = voxel_dict
		return output_record



	def lidar_to_histogram_features(self, lidar, crop=256, lidar_range=[28,28,28,28]):
		"""
		Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
		"""

		def splat_points(point_cloud):
			# 256 x 256 grid
			pixels_per_meter = 4
			hist_max_per_pixel = 5
			# x_meters_max = 28
			# y_meters_max = 28
			xbins = np.linspace(
				- lidar_range[3],
				lidar_range[2],
				(lidar_range[2]+lidar_range[3])* pixels_per_meter + 1,
			)
			ybins = np.linspace(-lidar_range[0], lidar_range[1], (lidar_range[0]+lidar_range[1]) * pixels_per_meter + 1)
			hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
			hist[hist > hist_max_per_pixel] = hist_max_per_pixel
			overhead_splat = hist / hist_max_per_pixel
			return overhead_splat

		below = lidar[lidar[..., 2] <= -1.45]
		above = lidar[lidar[..., 2] > -1.45]
		below_features = splat_points(below)
		above_features = splat_points(above)
		total_features = below_features + above_features
		features = np.stack([below_features, above_features, total_features], axis=-1)
		features = np.transpose(features, (2, 0, 1)).astype(np.float32)
		return features

	def collate_batch_infer_perception(self, car_data: list, rsu_data: list) -> dict:
		'''
		Re-collate a batch
		'''

		output_dict = {
            "lidar_pose": [],
            "voxel_features": [],
            "voxel_num_points": [],
            "voxel_coords": [],
            "lidar_original": [],
            "detmap_pose": [],
            "record_len": [],
			"target": [],
		}
		
		count = 0
		for j in range(len(car_data)):
			output_dict["record_len"].append(len(car_data)+len(rsu_data))
			output_dict["target"].append(car_data[j]['target_point'].unsqueeze(0).float())

			# Set j-th car as the ego-car.
			output_dict["lidar_original"].append(torch.from_numpy(car_data[j]['lidar_original']).unsqueeze(0))

                    
			output_dict["lidar_pose"].append(torch.from_numpy(car_data[j]['lidar_pose']).unsqueeze(0).cuda().float())
			output_dict["detmap_pose"].append(torch.from_numpy(car_data[j]['detmap_pose']).unsqueeze(0).cuda().float())
			count += 1
			for i in range(len(car_data)):
				if i==j:
					continue
				output_dict["lidar_original"].append(torch.from_numpy(car_data[i]['lidar_original']).unsqueeze(0))
				output_dict["lidar_pose"].append(torch.from_numpy(car_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(car_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
				count += 1
			for i in range(len(rsu_data)):
				output_dict["lidar_original"].append(torch.from_numpy(rsu_data[i]['lidar_original']).unsqueeze(0))
						
				output_dict["lidar_pose"].append(torch.from_numpy(rsu_data[i]['lidar_pose']).unsqueeze(0).cuda().float())
				output_dict["detmap_pose"].append(torch.from_numpy(rsu_data[i]['detmap_pose']).unsqueeze(0).cuda().float())
				count += 1
		for key in ["target", "lidar_pose", "detmap_pose" , "lidar_original"]:  # 
			output_dict[key] = torch.cat(output_dict[key], dim=0)
		
		output_dict["record_len"] = torch.from_numpy(np.array(output_dict["record_len"]))

		return output_dict
