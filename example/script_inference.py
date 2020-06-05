# import open3d as o3d
import torch

from ml3d import Predictor
from ml3d.util import Config

config_file = 'ml3d/config/semantic_segmentation/randlanet_semantickitti.py'
checkpoint_file = 'ml3d/checkpoint/randlanet_semantickitti.pth'
pointcloud_file = 'ml3d/datasets/example_data'


cfg         = Config.load_from_file(config_file)

device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

predictor   = Predictor(cfg, checkpoint_file, device=device)

#data        = o3d.io.read_point_cloud(pointcloud_file)
data        = cloud = 1000*torch.randn(1, 2**16, d_in).to(device)

result      = predictor(data)
o3d.visualization.draw()
