import os
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC

from loaders.nuscenes_dataset import get_nu_radar
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

import os, cv2, matplotlib.pyplot as plt
import numpy as np 
import torch
from mmdet3d.core.points import get_points_type

# 由于pypcd在Python 3.8中有兼容性问题，直接使用纯Python解析
PYPCD_AVAILABLE = False
print("Info: Using pure Python parsing for .pcd files.")


def load_pcd_points(pcd_path):
    """
    加载.pcd文件，支持pypcd和纯Python解析两种方式
    
    优先级：
    1. pypcd（如果已安装）
    2. 纯Python解析（备选）
    
    Args:
        pcd_path: .pcd文件路径
    
    Returns:
        points: (N, 7) numpy array
                [x, y, z, vx, vy, RCS, timestamp_offset]
    """
    if PYPCD_AVAILABLE:
        return _load_pcd_with_pypcd(pcd_path)
    else:
        return _load_pcd_with_python(pcd_path)


def _load_pcd_with_pypcd(pcd_path):
    """
    使用pypcd加载.pcd文件（推荐方式）
    
    支持两种格式：
    1. 完整格式：x, y, z, vx, vy, RCS (6字段)
    2. 当前格式：x, y, z, intensity (4字段)
    """
    pc = pypcd_module.PointCloud.from_path(pcd_path)
    N = pc.pc_data.shape[0]
    points = np.zeros((N, 7), dtype=np.float32)
    
    # 基础字段（必须）
    points[:, 0] = pc.pc_data['x']
    points[:, 1] = pc.pc_data['y']
    points[:, 2] = pc.pc_data['z']
    
    # 可选字段
    fields = pc.fields
    if 'vx' in fields:
        points[:, 3] = pc.pc_data['vx']
    if 'vy' in fields:
        points[:, 4] = pc.pc_data['vy']
    if 'RCS' in fields:
        points[:, 5] = pc.pc_data['RCS']
    elif 'intensity' in fields:
        points[:, 5] = pc.pc_data['intensity']  # 使用intensity作为RCS
    
    # timestamp offset默认为0
    points[:, 6] = 0
    
    return points


def _load_pcd_with_python(pcd_path):
    """
    纯Python解析.pcd文件（备选方式）
    适用于ASCII格式的.pcd文件
    """
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    
    # 解析头部
    fields = []
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('FIELDS'):
            fields = line.strip().split()[1:]
        elif line.startswith('DATA'):
            data_start = i + 1
            break
    
    # 解析数据
    data_list = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) >= len(fields):
            data_list.append([float(p) for p in parts])
    
    if len(data_list) == 0:
        raise ValueError(f"No data found in {pcd_path}")
    
    data = np.array(data_list, dtype=np.float32)
    N = data.shape[0]
    points = np.zeros((N, 7), dtype=np.float32)
    
    # 映射字段
    field_map = {
        'x': 0, 'y': 1, 'z': 2,
        'vx': 3, 'vy': 4,
        'RCS': 5, 'intensity': 5
    }
    
    for i, field in enumerate(fields):
        if field in field_map:
            points[:, field_map[field]] = data[:, i]
    
    return points


def load_npy_points(npy_path):
    """
    加载.npy文件（为未来准备）
    
    期望格式：(N, 6) 或 (N, 7) numpy array
              [x, y, z, vx, vy, RCS] 或 [x, y, z, vx, vy, RCS, timestamp]
    """
    points = np.load(npy_path)
    
    if points.shape[1] == 6:
        # 添加timestamp列
        timestamp = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, timestamp], axis=1)
    elif points.shape[1] == 7:
        # 已经包含timestamp
        pass
    else:
        raise ValueError(f"Expected 6 or 7 columns, got {points.shape[1]} columns")
    
    return points.astype(np.float32)


def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFuture(object):
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        # previous sweeps
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        # future sweeps
        if len(results['sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

from mmdet3d.core.points import BasePoints, get_points_type

@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@PIPELINES.register_module()
class LoadVoDPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 grid_config,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        self.downsample = 1
        self.grid_config = grid_config


    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        # depth_map = torch.zeros((height, width), dtype=torch.float32)
        # coor = torch.round(points[:, :2] / self.downsample)
        depth_map = np.zeros((height, width), dtype=np.float32)
        coor = np.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])

        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = np.ones(coor.shape[0], dtype=np.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        # coor = coor.to(torch.long)
        coor = coor.astype(np.int64)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        lidar2img_rt = results['lidar2img']
        img = results['img']
        num_points = points.shape[0]
        pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T

        # cam_points is Tensor of Nx4 whose last column is 1
        # transform camera coordinate to image coordinate
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]


        depth_map = self.points2depthmap(pts_2d.astype(np.float32), img.shape[0],
                                            img.shape[1])
        results['gt_depth'] = depth_map

        results['img_fields'].append('gt_depth')

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


    def depth_to_color(self, depth):
        cmap = plt.cm.jet
        #depth = np.expand_dims(depth, axis=2)
        
        d_min = np.min(depth)
        d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

    def image_draw(self, img, filename=None):
        # imgs = imgs.flatten(0,1).permute(0,2,3,1).contiguous()
        image = np.asarray(img, dtype=np.uint8)                        
        if filename: 
            img_path = os.path.join('./visual_outputs/vod/gt_depth', filename+'_cam'+'.png')                
            cv2.imwrite(img_path, image)


import torch
from pyquaternion import Quaternion

@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        img = results['img'][0]
        depth_map_list = []
        lidar2imgs = results['lidar2img'][:6]
        for lidar2img in lidar2imgs:
            lidar2img = torch.from_numpy(lidar2img).to(torch.float32)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)

            depth_map = self.points2depthmap(points_img, img.shape[0],
                                             img.shape[1])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class RadarPointToMultiViewDepth(object):
    
    def __init__(self, grid_config, downsample=1, test_mode=False):
        self.downsample = downsample
        self.grid_config = grid_config
        self.test_mode = test_mode

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        rcs_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        RCS = points[:, 3]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, RCS, ranks = coor[sort], depth[sort], RCS[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth, RCS = coor[kept2], depth[kept2], RCS[kept2]
        coor = coor.to(torch.long)

        depth_map[:, coor[:, 0]] = depth
        rcs_map[:, coor[:, 0]] = RCS
        return depth_map, rcs_map

    def load_offline(self, results):
        points_radar_ms = results['radar_points']
        img = results['img'][0]
        depth_map_list, rcs_map_list = [], []
        for i, points_radar in enumerate(points_radar_ms):
            lidar2imgs = results['lidar2img'][i*6:(i+1)*6]
            for lidar2img in lidar2imgs:
                lidar2img = torch.from_numpy(lidar2img).to(torch.float32)
                points_img = points_radar.tensor[:, :3].matmul(
                    lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                points_img = torch.cat(
                    [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3], points_radar.tensor[:, 3:4]],
                    1)

                depth_map, rcs_map = self.points2depthmap(points_img, img.shape[0],
                                                img.shape[1])
                depth_map_list.append(depth_map)
                rcs_map_list.append(rcs_map)
        depth_map = torch.stack(depth_map_list)
        rcs_map = torch.stack(rcs_map_list)
        results['radar_depth'] = depth_map
        results['radar_rcs'] = rcs_map
        return results

    def load_online(self, results):
        points_radar = results['radar_points'][0]
        img = results['img'][0]
        depth_map_list, rcs_map_list = [], []
        lidar2imgs = results['lidar2img'][:6]
        for lidar2img in lidar2imgs:
            lidar2img = torch.from_numpy(lidar2img).to(torch.float32)
            points_img = points_radar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3], points_radar.tensor[:, 3:4]],
                1)

            depth_map, rcs_map = self.points2depthmap(points_img, img.shape[0],
                                            img.shape[1])
            # 将当前帧复制8次（模拟8个时间帧）
            for _ in range(8):
                depth_map_list.append(depth_map)
                rcs_map_list.append(rcs_map)
        depth_map = torch.stack(depth_map_list)
        rcs_map = torch.stack(rcs_map_list)
        results['radar_depth'] = depth_map
        results['radar_rcs'] = rcs_map
        return results

    def __call__(self, results):

        return self.load_offline(results)
        
    
@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])             
                interval = np.random.randint(min_interval, max_interval + 1)
                
                # interval = 6  # here
                
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    intrinsic = sweep[sensor]['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    results['intrinsics'].append(viewpad)
        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval == 6

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    intrinsic = sweep[sensor]['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    results['intrinsics'].append(viewpad)

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results
        
        return self.load_offline(results)
    
@PIPELINES.register_module()
class Loadnuradarpoints(object):
    """Load radar Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 num_sweeps=5,
                 norm_time=False,
                 filter=True,
                 file_client_args=dict(backend='disk')):
        
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH', 'RADAR']
        
        self.get_nu_radar = get_nu_radar

        self.coord_type = coord_type
        self.num_sweeps = num_sweeps
        self.norm_time = norm_time
        self.filter = filter
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_own_radar(self, pts_filename):
        """
        加载自定义数据集的radar点云（支持.pcd和.npy格式）
        
        Args:
            pts_filename: radar文件路径
        
        Returns:
            points: torch.Tensor, shape (N, 7)
        """
        # 检查文件扩展名
        if pts_filename.endswith('.pcd'):
            points = load_pcd_points(pts_filename)
        elif pts_filename.endswith('.npy'):
            points = load_npy_points(pts_filename)
        else:
            # 默认按.bin格式处理（nuScenes格式）
            raise ValueError(f"Unsupported file format: {pts_filename}")
        
        # 转换为torch.Tensor
        points = torch.from_numpy(points).float()
        
        return points
    
    def __call__(self, results,):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        # 检查是否是自定义数据集（通过pts_filename判断）
        if 'pts_filename' in results and (results['pts_filename'].endswith('.pcd') or 
                                          results['pts_filename'].endswith('.npy')):
            # 自定义数据集：直接加载.pcd或.npy文件
            points = self._load_own_radar(results['pts_filename'])
            
            # 格式化为 (N, 7): [x, y, z, vx, vy, RCS, timestamp]
            # 保持与nuScenes相同的格式
            if self.coord_type == 'RADAR':
                coord_type = 'LIDAR'
            else:
                coord_type = self.coord_type
            
            points_class = get_points_type(coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None)
            
            results['radar_points'] = [points]
            results['radar_tokens'] = [[]]  # 空token列表
        else:
            # nuScenes数据集：使用原有逻辑
            points, radar_tokens, times = self.get_nu_radar(results['sample_idx'], True, self.num_sweeps, filter=self.filter)
            points = torch.cat([points, times], dim=0)
            points[2 , :] = 0 #-0.15

            points = points[[0,1,2,5,8,9,18], :].transpose(0, 1).contiguous()

            if self.coord_type == 'RADAR':
                coord_type = 'LIDAR'
            else:
                coord_type = self.coord_type
            points_class = get_points_type(coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None,)
            results['radar_points'] = [points]
            results['radar_tokens'] = [radar_tokens]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'num_sweeps={self.num_sweeps}, '
        return repr_str
    
@PIPELINES.register_module()
class LoadradarpointsFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 filter=True,
                 num_aggr_sweeps=6,
                 coord_type='RADAR',
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.test_mode = test_mode
        self.train_interval = [4, 8]
        self.test_interval = 6
        self.filter = filter
        self.num_aggr_sweeps = num_aggr_sweeps
        self.coord_type = coord_type
        self.get_nu_radar = get_nu_radar


    def load_offline(self, results):
        rad_types = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                points, radar_tokens, times = self.get_nu_radar(results['sample_idx'], True, self.num_aggr_sweeps, filter=self.filter)
                points = torch.cat([points, times], dim=0)
                points[2 , :] = 0 #-0.15
                points = points[[0,1,2,5,8,9,18], :].transpose(0, 1).contiguous()

                if self.coord_type == 'RADAR':
                    coord_type = 'LIDAR'
                points_class = get_points_type(coord_type)
                points = points_class(
                    points, points_dim=points.shape[-1], attribute_dims=None,)
                results['radar_points'].append(points)
                results['radar_tokens'].append(radar_tokens)
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(rad_types)+len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                points, radar_tokens, times = self.get_nu_radar(results['sample_idx'], True, self.num_aggr_sweeps, filter=self.filter, radar_sample_rec=sweep)
                points = torch.cat([points, times], dim=0)
                points[2 , :] = 0 #-0.15
                points = points[[0,1,2,5,8,9,18], :].transpose(0, 1).contiguous()
                if self.coord_type == 'RADAR':
                    coord_type = 'LIDAR'
                points_class = get_points_type(coord_type)
                points = points_class(
                    points, points_dim=points.shape[-1], attribute_dims=None,)
                results['radar_points'].append(points)
                results['radar_tokens'].append(radar_tokens)
        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval == 6

        rad_types = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                points, radar_tokens, times = self.get_nu_radar(results['sample_idx'], True, self.num_aggr_sweeps, filter=self.filter)
                points = torch.cat([points, times], dim=0)
                points[2 , :] = 0 #-0.15
                points = points[[0,1,2,5,8,9,18], :].transpose(0, 1).contiguous()
 
                if self.coord_type == 'RADAR':
                    coord_type = 'LIDAR'
                points_class = get_points_type(coord_type)
                points = points_class(
                    points, points_dim=points.shape[-1], attribute_dims=None,)

                results['radar_points'].append(points)
                results['radar_tokens'].append(radar_tokens)
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(rad_types)+len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                points, radar_tokens, times = self.get_nu_radar(results['sample_idx'], True, self.num_aggr_sweeps, filter=self.filter, radar_sample_rec=sweep)
                points = torch.cat([points, times], dim=0)
                points[2 , :] = 0 #-0.15
                points = points[[0,1,2,5,8,9,18], :].transpose(0, 1).contiguous()

                if self.coord_type == 'RADAR':
                    coord_type = 'LIDAR'
                points_class = get_points_type(coord_type)
                points = points_class(
                    points, points_dim=points.shape[-1], attribute_dims=None,)
                results['radar_points'].append(points)
                results['radar_tokens'].append(radar_tokens)

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results
        
        return self.load_offline(results)
