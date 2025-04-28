'''
File: edgecloud/scripts/cloud_slam.py
Project: Udf-Edge-Cloud SLAM
Created Date: 2024/08/26
Author: Luoxj
Email : ---
Summary:
'''
import sys
import argparse
import time
from attr import NOTHING
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as spR
from tqdm import tqdm
import os
sys.path.append(__file__)
import glob
# ROS # `source /opt/ros/noetic/setup.bash`
import rospy
import actionlib
from cv_bridge import CvBridge

# DROID-SLAM
sys.path.append('/data/xjluo/DROID-SLAM/droid_slam')
sys.path.append('/data/xjluo/DROID-SLAM')
import droid_backends
from lietorch import SE3

# custom
# from XX import ActCloudSlam
import torch.nn.functional as F
from droid_modified import Droid_mod

from util_conversion_dev import construct_covis_graph, downsample_voxel, extract_data, data_to_message
from downsample_feat_match import downsample_feat_match, downsample_feat_match_online

from msg_action.msg import CloudSlamAction, CloudSlamResult

#cap-udf
sys.path.append('/data/xjluo/cloud_udf/src/edgecloud/scripts/models/')
#from models.dataset_gpu import Dataset
from models.dataset_cpu import Dataset
#from models.dataset import Dataset
from models.chebykan_network import CAPUDFNetwork1
from pyhocon import ConfigFactory
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
import math
import paramiko  
from scp import SCPClient
import rosbag
from msg_action.msg import MapPoint
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from msg_action.msg import CloudMap
from tools.logger import print_log
import matplotlib.pyplot as plt

import open3d as o3d
from pyntcloud import PyntCloud
import time
import shutil 
torch.set_num_threads(14)  # 根据CPU核心数调整[3,5](@ref)

def wait_until_folder_non_empty(target_dir, check_interval=5):
    while True:
        try:
            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                if os.listdir(target_dir):  # 检测文件夹是否非空[3,7](@ref)
                    print(f"检测到 {target_dir} 非空，跳出循环")
                    break
                else:
                    print(f"云端SLAM3R场景重建中，等待 {check_interval} 秒后重试...")
            else:
                print(f"路径 {target_dir} 不存在或不是目录")
            time.sleep(check_interval)  # 设置检测间隔
        except PermissionError:
            print("权限不足，无法访问目标路径")
            break
        except KeyboardInterrupt:
            print("用户终止检测")
            break
def terminate_no_traj_filled(droid_):
    """ terminate the visualization process, return poses [t, q] """
    # del droid_.frontend

    torch.cuda.empty_cache()
    print("#" * 32)
    droid_.backend(7)

    torch.cuda.empty_cache()
    print("#" * 32)
    ijw = droid_.backend(12)
    return ijw

#一个图像生成器，用于处理图像序列并返回校准后的图像和内参
def image_calib_stream(sequence_image, calib, stride, image_size=[240, 320],  timestamps=None, save_path=None):
    """ image generator """
    fx, fy, cx, cy = calib[:4]
    bridge = CvBridge()

    for t, image_i in enumerate(sequence_image[::stride]):
        image = bridge.imgmsg_to_cv2(image_i, "bgr8")
        image = cv2.resize(image, (image_size[1], image_size[0]))

        if save_path and timestamps:
            timestamp = timestamps[t]
            save_image_path = os.path.join(save_path, f"image_{timestamp}.png")
            cv2.imwrite(save_image_path, image)

        image = torch.from_numpy(image).permute(2,0,1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0] *= image.shape[2] / 640.0
        intrinsics[1] *= image.shape[1] / 480.0
        intrinsics[2] *= image.shape[2] / 640.0
        intrinsics[3] *= image.shape[1] / 480.0

        yield t, image[None], intrinsics

def put_pth(pth, remote_path, file_path="/data/xjluo/cloud_udf/trans_work"):
    host = "10.23.11.18"  
    port = 22  
    username = "birl"  
    password = "birl123"  
 
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=15.0)
    local_path = file_path + "/" + pth
    try:
        scpclient.put(local_path, remote_path)
    except FileNotFoundError as e:
        print(e)
        print("The System Cannot Find The Specified File" + local_path)
    else:
        print("Cloud Data Successfully Downloaded To The Edge End")
    ssh_client.close()

class Cloud_Slam:
    def __init__(self,conf_path):
        
        #cap-udf
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # 启用CuDNN加速
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()
        
        self.conf = ConfigFactory.parse_string(conf_text)
        
        self.iter_step = 0
        self.old_udf = self.conf['general.old_udf']
        self.new_udf = self.conf['general.new_udf']
        self.step1_maxiter = self.conf.get_int('train.step1_maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.df_filter = self.conf.get_float('train.df_filter')

        self.ChamferDisL1 = ChamferDistanceL1().cuda()
        self.ChamferDisL2 = ChamferDistanceL2().cuda()
        
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.model_list = []
        self.writer = None
        
        self.udf_network = CAPUDFNetwork1().to(self.device)
        
        self.optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.udf_network.parameters(), lr=self.learning_rate,weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        #cap-udf

        parser = argparse.ArgumentParser()
        parser.add_argument("--datapath")
        parser.add_argument("--weights", default="/data/xjluo/DROID-SLAM/droid.pth")
        parser.add_argument("--buffer", type=int, default=512)
        parser.add_argument("--image_size", default=[240, 320])
        parser.add_argument("--disable_vis", action="store_true", default=True)

        parser.add_argument("--beta", type=float, default=0.6)
        parser.add_argument("--filter_thresh", type=float, default=1.75)
        parser.add_argument("--warmup", type=int, default=12)
        parser.add_argument("--keyframe_thresh", type=float, default=2.25)
        parser.add_argument("--frontend_thresh", type=float, default=12.0)
        parser.add_argument("--frontend_window", type=int, default=25)
        parser.add_argument("--frontend_radius", type=int, default=2)
        parser.add_argument("--frontend_nms", type=int, default=1)

        parser.add_argument("--backend_thresh", type=float, default=15.0)
        parser.add_argument("--backend_radius", type=int, default=2)
        parser.add_argument("--backend_nms", type=int, default=3)

        parser.add_argument("--calib", type=str, help="path to calibration file")
        parser.add_argument("--stride", type=int, default=1)
        self.args = parser.parse_args()
        self.args.stereo = False

        ### SLAM ### 串行处理地面端任务
        torch.multiprocessing.set_start_method('spawn')
        print(self.args)
        self.slam = Droid_mod(self.args, is_all_kf=True)
        time.sleep(1)#5

        # ROS node
        rospy.init_node('cloud')

        self.server = actionlib.SimpleActionServer("/cloud_slam", CloudSlamAction, self.slam_run, False)
        self.server.start()
        print("Server started...")

    def slam_run(self, goal):
        time_all_s = time.time()
        rospy.loginfo("Got the sequence! Lol")
        slam = self.slam
        sequence_image_edge = goal.sequence.images
        len_sequence = len(sequence_image_edge)
        tstamps_kf = goal.sequence.timestamps[::self.args.stride]
        camera = goal.sequence.camera
        K = camera.K
        intrinsics = [K[0], K[4], K[2], K[5]]

        cloud_time = time.time()
        for (t, image, cuda_intrinsics) in tqdm(image_calib_stream(sequence_image_edge, intrinsics, \
            self.args.stride, image_size=[240, 320], timestamps=tstamps_kf, save_path="/data/xjluo/SLAM3R/data/Replica_demo/data")):
            slam.track(t, image, intrinsics=cuda_intrinsics)
        
        txt_file_path = os.path.join("/data/xjluo/SLAM3R/data/Replica_demo/data", f"Image_data_isok.txt")  # 定义txt路径[3,5](@ref)
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:  # 使用with自动关闭文件[2,6](@ref)
                # 写入内容示例（可自定义）
                f.write("SLAM3R Dataset is ok!")  # 图像路径
        except IOError as e:
            print(f"Error writing txt file: {e}")  # 异常处理

        graph = terminate_no_traj_filled(slam)
        print("#"*20+"Cloud took {:.2f} seconds".format(time.time()-cloud_time))

        idx_keyframe, tstamps_keyframe, images_keyframe, poses_keyframe, \
            depths_keyframe, intrinsics_keyframe, intrisics_fullsize, index_xyz_duv, colors, graph_ijw \
            = extract_data(slam, self.args, graph, intrinsics, tstamps_kf)
        ### construct mappoint covisibility graph ###
        if (self.new_udf):
            np.save('/data/xjluo/cloud_udf/trans_work/images_keyframe.npy', images_keyframe)
            np.save('/data/xjluo/cloud_udf/trans_work/idx_keyframe.npy', idx_keyframe)
            np.save('/data/xjluo/cloud_udf/trans_work/poses_keyframe.npy', poses_keyframe)
            np.save('/data/xjluo/cloud_udf/trans_work/depths_keyframe.npy',depths_keyframe)
            np.save('/data/xjluo/cloud_udf/trans_work/intrinsics_keyframe.npy', intrinsics_keyframe)
            np.save('/data/xjluo/cloud_udf/trans_work/intrisics_fullsize.npy', intrisics_fullsize)
            np.save('/data/xjluo/cloud_udf/trans_work/tstamps_keyframe.npy', tstamps_keyframe)
            put_pth("idx_keyframe.npy", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others")
            put_pth("poses_keyframe.npy", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others")
            put_pth("depths_keyframe.npy", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others")
            put_pth("intrinsics_keyframe.npy", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others")
            put_pth("intrisics_fullsize.npy", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others")
            put_pth("tstamps_keyframe.npy", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_others")
        star_time = time.time()
        wait_until_folder_non_empty('/data/xjluo/SLAM3R/results/Replica_demo', check_interval=1)
        end_time = time.time()
        duration_time = end_time - star_time
        print(f"SLAM3R Restruction Time: {duration_time:.2f} 秒")
        pointcloud = o3d.io.read_point_cloud('/data/xjluo/SLAM3R/results/Replica_demo/Replica_demo_data_recon.ply')
        pointcloud = np.asarray(pointcloud.points).astype(np.float16)
        #cap-udf
        if (self.old_udf or self.new_udf):
            star_time = time.time()
            self.dataset = Dataset(pointcloud)
            end_time = time.time()
            duration_time = end_time - star_time
            print(f"Load data Time: {duration_time:.2f} 秒")
            self.train()
        #cap-udf
        pointcloud_droidslam_map = index_xyz_duv[:,1:4]
        goal.downsample_ratio = 0.1
        #pc_v = downsample_voxel(pointcloud, do_ratio=True, ratio=goal.downsample_ratio, tolerance_ratio=0.05)
        pc_v = downsample_voxel(pointcloud_droidslam_map, do_ratio=True, ratio=goal.downsample_ratio, tolerance_ratio=0.05)
        print("Num of downsample:",pc_v.shape)
        os.remove('/data/xjluo/SLAM3R/results/Replica_demo/Replica_demo_data_recon.ply')
        if (self.new_udf):
            put_pth("test.pth", "/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_pth")
            print("Test edge front map mnid: ", goal.sequence.edge_front_map_mnid)
            print("Test edge back map mnid: ", goal.sequence.edge_back_map_mnid)

            result = CloudSlamResult()
            pub_cloud_map = CloudMap()
            pub_cloud_map.header = Header()
            ros_mappoints = []
            for i in range(pc_v.shape[0]):
                # point
                ros_point = Point(x = pc_v[i][0], y = pc_v[i][1], z = pc_v[i][2])
                ros_mappoint = MapPoint()
                ros_mappoint.point = ros_point
                ros_mappoint.mnId = i
                ros_mappoints.append(ros_mappoint)
                pass
            pub_cloud_map.map_points = ros_mappoints
            result.map = pub_cloud_map
            result.map.edge_front_map_mnid = goal.sequence.edge_front_map_mnid
            result.map.edge_back_map_mnid  = goal.sequence.edge_back_map_mnid
    
        self.server.set_succeeded(result)
        time_all_e = time.time()
        time_all = time_all_e - time_all_s
        print("cloud_time:{}".format(time_all))

        # ### clear
        self.slam = Droid_mod(self.args, is_all_kf=True)

        if (False):
            os.remove('idx_keyframe.npy')
            os.remove('poses_keyframe.npy')
            os.remove('depths_keyframe.npy')
            os.remove('intrinsics_keyframe.npy')
            os.remove('intrisics_fullsize.npy')
            os.remove('tstamps_keyframe.npy')
        time.sleep(1)
        self.iter_step = 0
        
    def train(self):
        torch.cuda.empty_cache()
        print("#" * 32)
        batch_size = self.batch_size
        ratios = []
        losses = []  # 用于保存每步的损失值
        time_list = []  # 用于记录每次迭代的时间
        for iter_i in tqdm(range(self.iter_step, self.step1_maxiter)):
            # start_time = time.time()  # 记录开始时间
            self.update_learning_rate(self.iter_step)

            if self.iter_step < self.step1_maxiter:
                points, samples, point_gt = self.dataset.get_train_data(batch_size) #从npz里拿东西6
                
                samples.requires_grad = True
                gradients_sample = self.udf_network.gradient(samples).squeeze() # 5000x3
                udf_sample = self.udf_network.udf(samples)                      # 5000x1
                grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
                sample_moved = samples - grad_norm * udf_sample                 # 5000x3

                loss_cd = self.ChamferDisL1(points.unsqueeze(0), sample_moved.unsqueeze(0))
            
                loss = loss_cd   
                # 记录损失
                losses.append(loss_cd.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.iter_step += 1
                if self.iter_step % self.report_freq == 0:
                    print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss_cd, self.optimizer.param_groups[0]['lr']))
                if self.iter_step == self.step1_maxiter:
                    self.save_checkpoint()           

    def update_learning_rate(self, iter_step):

        warn_up = self.warm_up_end
        max_iter = self.step1_maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr

    def save_checkpoint(self):
        checkpoint = {
            'udf_network_fine': self.udf_network.state_dict(),
        }
        torch.save(checkpoint, '/data/xjluo/cloud_udf/trans_work/test.pth')
        print("the model is saved successfully")
        
    #cap-udf
    def gen_extra_pointcloud(self, low_range):

        res = []
        num_points = 1000000
        gen_nums = 0

        while gen_nums < num_points:
            
            points, samples, point_gt = self.dataset.get_train_data(5000)
            offsets = samples - points
            std = torch.std(offsets)
            std = torch.std(points)
            
            extra_std = std * low_range
            rands = torch.normal(0.0, extra_std, size=points.shape)   
            samples = points + torch.tensor(rands).cuda().float()

            samples.requires_grad = True
            gradients_sample = self.udf_network.gradient(samples).squeeze() # 5000x3
            udf_sample = self.udf_network.udf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * udf_sample                 # 5000x3

            index = udf_sample < self.df_filter
            index = index.squeeze(1)
            sample_moved = sample_moved[index]
            
            gen_nums += sample_moved.shape[0]

            res.append(sample_moved.detach().cpu().numpy())
        scale = self.dataset.get_scale()
        res = np.concatenate(res) * scale
        res = res[:num_points]
        return res            
    #cap-udf

    def slam_deinit(self,):
        ### deinit ###
        del self.slam

    
if __name__ == "__main__":
    star_time = time.time()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    cloudslalm = Cloud_Slam('/data/xjluo/cloud_udf/src/edgecloud/scripts/confs/base.conf')
    rospy.spin()
    
