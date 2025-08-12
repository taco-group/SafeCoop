# late fusion dataset
import random
import math
from collections import OrderedDict
import cv2 as cv
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)



def getLateclassFusionDataset(cls):
    """
    cls: the BaseDataset or父类数据集, 负责一些基础接口，如:
         - retrieve_base_data()
         - generate_object_center_single()
         - self.post_processor
         - self.pre_processor
         - self.selector (如果用了 heterogeneous 配置)
         等等
    """
    class LateclassFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            # 是否启用异构学习(例如只选择某些Agent用lidar，某些Agent用camera)
            self.heterogeneous = False
            if "heter" in params:
                self.heterogeneous = True

            # 是否为多类别
            self.multiclass = params["model"]["args"].get("multi_class", False)

            # 根据需要，可在这里给定多类别的类别 ID 列表
            # 比如 [0, 1, 3] 分别对应 car / pedestrian / cyclist 等
            self.class_list = params.get("class_list", [0, 1, 3])
            # 若项目里您是通过 [ 'all', 0, 1, 3 ] 这种方式区分，也可自行调整

            # 用于可视化
            self.visualize = visualize
            self.train = train

        def __getitem__(self, idx):
            """
            训练阶段：随机选 1 个 CAV 做 late 监督(与LateFusionDataset一致)；
            测试/验证阶段：保留所有范围内 CAV 的信息。
            """
            base_data_dict = self.retrieve_base_data(idx)
            if self.train:
                reformat_data_dict = self.get_item_train(base_data_dict)
            else:
                reformat_data_dict = self.get_item_test(base_data_dict, idx)
            return reformat_data_dict

        def get_item_train(self, base_data_dict):
            """
            训练阶段的处理逻辑：通常是只抽取 1 个 CAV（含有 label），
            以减少内存开销、保持与单车训练类似。
            """
            from collections import OrderedDict
            processed_data_dict = OrderedDict()

            # 数据扰动（如果有）
            base_data_dict = self.add_noise_data_if_needed(base_data_dict)

            # 只随机抽取一个 CAV
            if not self.visualize:
                selected_cav_id, selected_cav_base = random.choice(
                    list(base_data_dict.items())
                )
            else:
                # 若要可视化，通常选 ego 做可视化
                selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]

            # 处理单个车辆（含多类别的 bbox）
            cav_processed = self.get_item_single_car(selected_cav_base)
            processed_data_dict["ego"] = cav_processed
            return processed_data_dict

        def get_item_test(self, base_data_dict, idx):
            """
            测试/验证阶段：保留所有在 comm_range 内的 CAV，都要 late fusion 的 label。
            """
            from collections import OrderedDict
            import math

            base_data_dict = self.add_noise_data_if_needed(base_data_dict)

            processed_data_dict = OrderedDict()
            ego_id, ego_pose = -1, None
            # 首先找到 ego
            for cav_id, cav_content in base_data_dict.items():
                if cav_content["ego"]:
                    ego_id = cav_id
                    ego_pose = cav_content["params"]["lidar_pose"]
                    ego_pose_clean = cav_content["params"]["lidar_pose_clean"]
                    break
            assert ego_id != -1

            cav_id_list = []
            for cav_id, cav_content in base_data_dict.items():
                distance = math.sqrt(
                    (cav_content["params"]["lidar_pose"][0] - ego_pose[0]) ** 2
                    + (cav_content["params"]["lidar_pose"][1] - ego_pose[1]) ** 2
                )
                if distance <= self.params["comm_range"]:
                    cav_id_list.append(cav_id)

            cav_id_list_newname = []
            for cav_id in cav_id_list:
                selected_cav_base = base_data_dict[cav_id]
                transformation_matrix = self.x1_to_x2(
                    selected_cav_base["params"]["lidar_pose"], ego_pose
                )
                transformation_matrix_clean = self.x1_to_x2(
                    selected_cav_base["params"]["lidar_pose_clean"], ego_pose_clean
                )
                cav_processed = self.get_item_single_car(selected_cav_base)
                cav_processed.update(
                    {
                        "transformation_matrix": transformation_matrix,
                        "transformation_matrix_clean": transformation_matrix_clean,
                    }
                )
                # 若是 ego 自身，就命名为 "ego"，否则保持 cav_id
                update_cav_key = "ego" if cav_id == ego_id else cav_id
                processed_data_dict[update_cav_key] = cav_processed
                cav_id_list_newname.append(update_cav_key)

            # heterogeneous 额外信息
            if self.heterogeneous:
                processed_data_dict["ego"]["idx"] = idx
                processed_data_dict["ego"]["cav_list"] = cav_id_list_newname

            return processed_data_dict

        def get_item_single_car(self, cav_base):
            """
            处理单辆车的信息，生成其多类别的 label、lidar 数据、camera 数据等等。
            """
            selected_cav_processed = {}

            # 1) 生成多类别或单类别目标框
            #   如果多类别，就将 cav_base 中属于各类的目标框分开存储/或一次性存 [num_class, max_box, 7]
            if self.multiclass:
                # 举例：将 class_list = [0,1,3] 三个类别分别解析
                # 最简单做法是：对 cav_base["params"]["lidar_pose_clean"] 调用多次 generate_object_center_single
                # 并把结果堆叠
                all_box_list, all_mask_list, all_ids_list = [], [], []
                for cls_id in self.class_list:
                    box_c, mask_c, ids_c = self.generate_object_center_single(
                        [cav_base],
                        cav_base["params"]["lidar_pose_clean"],
                        class_type=cls_id,  # 您可在 generate_object_center_single 里根据 class_type 做过滤
                    )
                    all_box_list.append(box_c)
                    all_mask_list.append(mask_c)
                    all_ids_list.append(ids_c)

                # 堆叠成 [num_class, max_box, 7] / [num_class, max_box]
                # 需注意每次 generate_object_center_single 返回的 max_box 数量可能不同,
                # 这里需统一补零或 slice 到相同维度(可参考已有Late/IntermediateFusion实现).
                object_bbx_center, object_bbx_mask = self.stack_multiclass_label(
                    all_box_list, all_mask_list
                )
                # object_ids 可以按类别各存一个 list，也可以只存 [num_class, ...]
                object_ids = all_ids_list  # 也可做特殊处理
            else:
                # 单类别情况下：直接一次即可
                object_bbx_center, object_bbx_mask, object_ids = (
                    self.generate_object_center_single(
                        [cav_base], cav_base["params"]["lidar_pose_clean"]
                    )
                )

            # 2) lidar 处理(或 camera)
            #   若需要 lidar，可做 voxelize -> self.pre_processor
            if self.load_lidar_file or self.visualize:
                lidar_np = cav_base["lidar_np"]
                # 一些基础处理，如 shuffle_points, mask_points_by_range, mask_ego_points 等
                lidar_np = self.basic_lidar_preprocess(lidar_np)
                # 数据增强(根据需要)
                lidar_np, object_bbx_center, object_bbx_mask = self.augment_if_needed(
                    lidar_np, object_bbx_center, object_bbx_mask
                )
                # 真正处理，如 voxelize/BEV projection
                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed["processed_lidar"] = processed_lidar

                if self.visualize:
                    selected_cav_processed["origin_lidar"] = lidar_np

            # 3) camera 处理
            if self.load_camera_file:
                # 类似 LateFusionDataset 中的逻辑
                camera_inputs = self.process_camera_data(cav_base)
                selected_cav_processed["image_inputs"] = camera_inputs

            # 4) 保存多类别框
            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                }
            )

            # 5) 生成 label，若多类别则也要多类别 label
            if self.multiclass:
                # 自行封装 post_processor.generate_label(...) 以支持 multi-class
                # 也可对每个类别分别调用
                label_dict = self.post_processor.generate_label_multiclass(
                    object_bbx_center,  # [num_class, max_box, 7]
                    self.anchor_box,
                    object_bbx_mask,    # [num_class, max_box]
                )
            else:
                label_dict = self.post_processor.generate_label(
                    object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
                )

            selected_cav_processed["label_dict"] = label_dict
            return selected_cav_processed

        ############################
        # collate_batch 相关处理  #
        ############################
        def collate_batch_train(self, batch):
            """
            训练集的 collate：
            由于本示例中 train 阶段只随机取了 1 个 CAV，直接按 batch 拼接即可。
            若您想要真正多 CAV 的 late 监督训练，则需参考 test collate 的思路。
            """
            import torch
            from collections import OrderedDict
            output_dict = {"ego": {}}

            object_bbx_center_list = []
            object_bbx_mask_list = []
            label_dict_list = []
            origin_lidar_list = []

            processed_lidar_list = []

            for item in batch:
                ego_data = item["ego"]
                object_bbx_center_list.append(ego_data["object_bbx_center"])
                object_bbx_mask_list.append(ego_data["object_bbx_mask"])
                label_dict_list.append(ego_data["label_dict"])

                if self.visualize and "origin_lidar" in ego_data:
                    origin_lidar_list.append(ego_data["origin_lidar"])

                if "processed_lidar" in ego_data:
                    processed_lidar_list.append(ego_data["processed_lidar"])

            # 转成 tensor
            object_bbx_center_torch = self.list_to_tensor(object_bbx_center_list)
            object_bbx_mask_torch = self.list_to_tensor(object_bbx_mask_list)

            # 多类别 label 的 collate (或单类别)
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)
            # 若使用 centerpoint, 还要再把 object_bbx_center_torch 等融合进 label_torch_dict
            label_torch_dict.update(
                {
                    "object_bbx_center": object_bbx_center_torch,
                    "object_bbx_mask": object_bbx_mask_torch,
                }
            )

            output_dict["ego"].update(
                {
                    "object_bbx_center": object_bbx_center_torch,
                    "object_bbx_mask": object_bbx_mask_torch,
                    "anchor_box": torch.from_numpy(self.anchor_box),
                    "label_dict": label_torch_dict,
                }
            )

            # lidar
            if len(processed_lidar_list) > 0:
                processed_lidar_torch_dict = self.pre_processor.collate_batch(
                    processed_lidar_list
                )
                output_dict["ego"]["processed_lidar"] = processed_lidar_torch_dict

            # camera
            if self.load_camera_file:
                # 类似 LateFusionDataset: 将 batch 里的 camera 信息按维度拼起来
                camera_inputs = self.collate_camera_inputs_train(batch)
                output_dict["ego"]["image_inputs"] = camera_inputs

            # visualization
            if self.visualize and len(origin_lidar_list) > 0:
                # 您可以根据需要 downsample
                origin_lidar_torch = self.list_to_tensor(origin_lidar_list)
                output_dict["ego"]["origin_lidar"] = origin_lidar_torch

            return output_dict

        def collate_batch_test(self, batch):
            """
            测试集（或验证集）的 collate：
            一般只支持 batch_size=1（尤其在多 CAV 的情况下），
            然后把每个 CAV 单独拿出来做 late 处理。
            """
            assert len(batch) == 1, "Test time batch_size must be 1 for late fusion!"
            batch = batch[0]

            output_dict = {}
            # heterogeneous
            if self.heterogeneous and "idx" in batch["ego"]:
                idx = batch["ego"]["idx"]
                cav_list = batch["ego"]["cav_list"]
                # 选择哪些 cav 用 lidar / camera
                # lidar_agent, camera_agent = self.selector.select_agent(idx)
                # ...

            # 收集并 collate
            if self.visualize:
                import copy
                projected_lidar_list = []

            for cav_id, cav_content in batch.items():
                output_dict[cav_id] = {}
                # 把 object_bbx_center/mask 变成 [1, ...]
                object_bbx_center = self.unsqueeze_to_batch(cav_content["object_bbx_center"])
                object_bbx_mask = self.unsqueeze_to_batch(cav_content["object_bbx_mask"])

                label_dict = self.post_processor.collate_batch([cav_content["label_dict"]])
                # centerpoint 需把 object_bbx_center/mask 再塞回 label_dict
                label_dict.update(
                    {
                        "object_bbx_center": object_bbx_center,
                        "object_bbx_mask": object_bbx_mask,
                    }
                )

                # lidar
                if "processed_lidar" in cav_content:
                    # 只有 1 个 cav 的 processed_lidar
                    processed_lidar_torch = self.pre_processor.collate_batch(
                        [cav_content["processed_lidar"]]
                    )
                    output_dict[cav_id]["processed_lidar"] = processed_lidar_torch

                # camera
                if self.load_camera_file and "image_inputs" in cav_content:
                    # 同理，只拼一个
                    cam_torch = self.collate_camera_inputs_test(cav_content)
                    output_dict[cav_id]["image_inputs"] = cam_torch

                # heterogeneous 可根据 cav_id 判断是否保留/剔除
                # if self.heterogeneous:
                #     pass

                # 保存变换矩阵
                output_dict[cav_id]["transformation_matrix"] = torch.from_numpy(
                    cav_content["transformation_matrix"]
                ).float()
                output_dict[cav_id]["transformation_matrix_clean"] = torch.from_numpy(
                    cav_content["transformation_matrix_clean"]
                ).float()

                # label + 其他信息
                output_dict[cav_id].update(
                    {
                        "object_bbx_center": object_bbx_center,
                        "object_bbx_mask": object_bbx_mask,
                        "label_dict": label_dict,
                        "anchor_box": self.anchor_box_torch,
                        "object_ids": cav_content["object_ids"],
                    }
                )

                if self.visualize and "origin_lidar" in cav_content:
                    output_dict[cav_id]["origin_lidar"] = torch.from_numpy(
                        cav_content["origin_lidar"]
                    )

            # 若需要把多 cav 的点云拼接到 ego 上做可视化，可以在这里做拼接
            return output_dict

        ######################################
        #          多类别后处理示例          #
        ######################################
        def post_process(self, data_dict, output_dict):
            """
            如果是多类别，就调用 self.post_process_multiclass，
            否则与普通 late fusion 相同。
            """
            if self.multiclass:
                # 返回 [List of pred_box], [List of score], [List of gt_box]，每个元素对应一个类别
                return self.post_process_multiclass(data_dict, output_dict)
            else:
                pred_box, pred_score = self.post_processor.post_process(data_dict, output_dict)
                gt_box = self.post_processor.generate_gt_bbx(data_dict)
                return pred_box, pred_score, gt_box

        def post_process_multiclass(self, data_dict, output_dict):
            """
            多类别的后处理，每个类别各跑一次 NMS 或类似处理，然后拼一起返回。
            """
            import copy

            # num_class = len(self.class_list)
            pred_box_tensor_list = []
            pred_score_list = []
            gt_box_tensor_list = []

            # 对每个类别独立后处理
            for i, cls_id in enumerate(self.class_list):
                # 1) 拷贝出仅包含该类别的数据
                data_dict_single, output_dict_single = self.split_single_class(
                    data_dict, output_dict, class_index=i
                )
                # 2) 跑后处理
                pred_box_tensor, pred_score = self.post_processor.post_process(
                    data_dict_single, output_dict_single
                )
                gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict_single)

                pred_box_tensor_list.append(pred_box_tensor)
                pred_score_list.append(pred_score)
                gt_box_tensor_list.append(gt_box_tensor)

            return pred_box_tensor_list, pred_score_list, gt_box_tensor_list

        ############################################
        # 下方放一些复用/简化方法(根据项目适配即可)  #
        ############################################
        def add_noise_data_if_needed(self, base_data_dict):
            """
            根据 self.params["noise_setting"] 等需求决定是否进行噪声扰动。
            这里直接调用已有的 add_noise_data_dict 或 add_noise_data_dict_asymmetric。
            """
            from opencood.utils.pose_utils import add_noise_data_dict
            # 如果想用非对称噪声，请自行替换
            return add_noise_data_dict(base_data_dict, self.params["noise_setting"])

        def basic_lidar_preprocess(self, lidar_np):
            """
            一些通用的点云预处理，如范围裁剪、shuffle、去除自车点等。
            """
            from opencood.utils.pcd_utils import (
                shuffle_points,
                mask_points_by_range,
                mask_ego_points,
            )
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
            lidar_np = mask_ego_points(lidar_np)
            return lidar_np

        def augment_if_needed(self, lidar_np, object_bbx_center, object_bbx_mask):
            """
            若 self.train 并且无需异构，可对点云/标签做数据增强。
            """
            if self.train and not self.heterogeneous:
                lidar_np, object_bbx_center, object_bbx_mask = self.augment(
                    lidar_np, object_bbx_center, object_bbx_mask
                )
            return lidar_np, object_bbx_center, object_bbx_mask

        def process_camera_data(self, cav_base):
            """
            将相机图像根据参数（分辨率缩放、裁剪、flip 等）做增广，并返回成一个 dict。
            可参考 LateFusionDataset / LSS 处理流程。
            """
            # 这里仅示例化简, 具体实现请参考原 LateFusionDataset 中的 get_item_single_car -> process_camera_data
            camera_data_list = cav_base["camera_data"]
            # ... 做增广与 transform ...
            camera_inputs = {"imgs": None, "rots": None, ...}
            return camera_inputs

        def collate_camera_inputs_train(self, batch):
            """
            将 train batch 里多帧图像按维度拼接，比如 [B, N, C, H, W]
            """
            # 略，参考 LateFusionDataset 的 collate_batch_train
            return {}

        def collate_camera_inputs_test(self, cav_content):
            """
            测试阶段只 collate 单个 cav
            """
            # 参考 LateFusionDataset 的 collate_batch_test
            return {}

        def stack_multiclass_label(self, box_list, mask_list):
            """
            输入是一个 list，每个元素是 (max_box, 7)/(max_box,),
            最终拼成 [num_class, max_box, 7] / [num_class, max_box]。
            若每个类别分配的 max_box 不同，需要先找最大值再做 padding。
            """
            import numpy as np
            num_class = len(box_list)
            max_box_counts = [b.shape[0] for b in box_list]
            M = max(max_box_counts) if max_box_counts else 0

            # 组合
            box_array = []
            mask_array = []
            for i in range(num_class):
                cur_box = box_list[i]
                cur_mask = mask_list[i]
                pad_size = M - cur_box.shape[0]
                if pad_size > 0:
                    # 在 0 处 padding
                    cur_box = np.concatenate(
                        [cur_box, np.zeros((pad_size, 7), dtype=cur_box.dtype)], axis=0
                    )
                    cur_mask = np.concatenate(
                        [cur_mask, np.zeros(pad_size, dtype=cur_mask.dtype)], axis=0
                    )
                box_array.append(cur_box[None, ...])   # [1, M, 7]
                mask_array.append(cur_mask[None, ...]) # [1, M]

            if len(box_array) == 0:
                # 说明没对象
                return np.zeros((0, 0, 7)), np.zeros((0, 0))

            box_array = np.concatenate(box_array, axis=0)   # [num_class, M, 7]
            mask_array = np.concatenate(mask_array, axis=0) # [num_class, M]
            return box_array, mask_array

        def split_single_class(self, data_dict, output_dict, class_index):
            """
            post_process_multiclass 用到：
            将 data_dict/output_dict 中多类别的 object_bbx_center/mask
            拆分出第 class_index 个类别的子数据，以便单独跑 NMS。
            """
            import copy
            data_dict_single = {"ego": {}}
            output_dict_single = {}

            # 遍历所有 cav (late fusion)
            for cav_id in data_dict.keys():
                cav_content = data_dict[cav_id]
                cav_output = output_dict[cav_id]

                # 如果 object_bbx_center 是 [num_class, M, 7]，mask 是 [num_class, M]
                # 拆分出 cav_idx = class_index 这一路
                single_box_center = cav_content["object_bbx_center"][class_index, ...]
                single_mask = cav_content["object_bbx_mask"][class_index, ...]
                # object_ids 如果是按类别存储的list，可按 class_index 取即可
                # 如果合并一起，需要自己额外做记录
                if isinstance(cav_content["object_ids"], list):
                    single_ids = cav_content["object_ids"][class_index]
                else:
                    single_ids = cav_content["object_ids"]  # 或者看具体储存方式

                # 类似地，对网络输出 cls_preds, reg_preds_multiclass 都要取第 class_index 路
                # 具体看原网络 forward 的输出 shape
                cls_preds_single = cav_output["cls_preds"][
                    :, class_index : class_index + 1, :, :
                ]  # e.g. [B,1,H,W]
                reg_preds_single = cav_output["reg_preds_multiclass"][
                    :, class_index, :, :
                ]  # [B,H,W,Nreg]

                # 构造新的 data_dict_single / output_dict_single
                data_dict_single[cav_id] = copy.deepcopy(cav_content)
                data_dict_single[cav_id]["object_bbx_center"] = single_box_center[None, ...]  # 保留一个 batch 维
                data_dict_single[cav_id]["object_bbx_mask"] = single_mask[None, ...]
                data_dict_single[cav_id]["object_ids"] = single_ids

                output_dict_single[cav_id] = copy.deepcopy(cav_output)
                output_dict_single[cav_id]["cls_preds"] = cls_preds_single
                output_dict_single[cav_id]["reg_preds"] = reg_preds_single

            return data_dict_single, output_dict_single

        ###################################################
        # 一些工具函数(和原 LateFusionDataset/中间类一致) #
        ###################################################
        def x1_to_x2(self, lidar_pose1, lidar_pose2):
            """
            位姿变换矩阵, 与 opencood.utils.transformation_utils.x1_to_x2 一致。
            """
            return x1_to_x2(lidar_pose1, lidar_pose2)

        def list_to_tensor(self, data_list):
            """
            简易把 list of np.array 变成 torch.Tensor, 做 batch 拼接用。
            """
            import numpy as np
            import torch
            if len(data_list) == 0:
                return None
            arr = np.stack(data_list, axis=0)
            return torch.from_numpy(arr)

        def unsqueeze_to_batch(self, arr):
            """
            如果 arr 是 np.ndarray，就转成 [1, ...]，再转成 torch。
            """
            import numpy as np
            import torch
            if isinstance(arr, np.ndarray):
                arr = arr[None, ...]  # 在前面加一个 batch 维
                arr = torch.from_numpy(arr)
            elif isinstance(arr, torch.Tensor) and arr.dim() == 2:
                # [M,7] -> [1,M,7]
                arr = arr.unsqueeze(0)
            return arr

    return LateMultiFusionDataset