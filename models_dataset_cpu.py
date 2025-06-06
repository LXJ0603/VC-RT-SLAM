import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import torch
torch.set_num_threads(14)  # 根据CPU核心数调整[3,5](@ref)

def search_nearest_point(point_batch, point_gt):
    distances = torch.cdist(point_batch, point_gt)
    dis_idx = torch.argmin(distances, dim=1)
    return dis_idx.cpu().numpy()  # 确保返回CPU上的NumPy数组[3,5](@ref)
def process_data(pointcloud):
    torch.cuda.empty_cache()
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60 #"//"向下取整 原始点云数量/60，再取整
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60 #取整后*60
    QUERY_EACH = 1000000//POINT_NUM_GT #？

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False) #从[0,原始数据点]中取POINT_NUM_GT个数，组成一维数组
    pointcloud = pointcloud[point_idx,:]
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0): #将数量拆成100组
        d = ptree.query(p,51) #ckdtree查询
        sigmas.append(d[0][:,-1])
    
    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []

    for i in range(QUERY_EACH):
        scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt) #*
        tt = tt.reshape(-1,POINT_NUM,3)

        sample_near_tmp = []
        for j in range(tt.shape[0]):
            nearest_idx = search_nearest_point(
                torch.tensor(tt[j]).float(), 
                torch.tensor(pointcloud).float()
            )
            nearest_points = pointcloud[nearest_idx]  # 此时nearest_idx是NumPy数组
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp) #*
        
    sample = np.asarray(sample) #*
    sample_near = np.asarray(sample_near) #*

    return pointcloud,sample,sample_near,shape_scale

class Dataset:
    def __init__(self, point_data):
        super(Dataset, self).__init__() 
        print('Load data: Begin')
        self.device = torch.device('cuda')

        point,sample,sample_near,self.scale = process_data(point_data) 

        
        self.point = np.asarray(sample_near.reshape(-1,3)) #-1表示行数自动计算，3表示3列 asarray将列表转换为数组
        self.sample = np.asarray(sample.reshape(-1,3))
        self.point_gt = np.asarray(point.reshape(-1,3)) #原始点云
        self.sample_points_num = self.sample.shape[0]-1

        self.object_bbox_min = np.array([np.min(self.point[:,0]), np.min(self.point[:,1]), np.min(self.point[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point[:,0]), np.max(self.point[:,1]), np.max(self.point[:,2])]) +0.05
        print('bd:',self.object_bbox_min,self.object_bbox_max)

        self.point = torch.from_numpy(self.point).to(self.device).float() #"torch.from_numpy()"将numpy转为tensor
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
        
        print('NP Load data: End')

    def get_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse  # for accelerating random choice operation
        points = self.point[index]
        sample = self.sample[index]
        return points, sample, self.point_gt

    def gen_new_data(self, tree):
        distance, index = tree.query(self.sample.detach().cpu().numpy(), 1)
        self.point_new = tree.data[index]
        self.point_new = torch.from_numpy(self.point_new).to(self.device).float()

    def get_scale(self):
        scale = self.scale
        return scale    
