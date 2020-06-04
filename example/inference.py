import open3d as o3d
from ml3d import Predictor
from ml3d.util import Config

config_file = 'config/semantic_segmentation/randlanet_semantickitti.py'
checkpoint_file = 'checkpoint/randlanet_semantickitti.pth'
pointcloud_file = 'datasets/example_data'


cfg         = Config.load_from_file(config_file)
predictor   = Predictor(cfg, checkpoint_file, device='cuda:0')
data        = o3d.io.read_point_cloud(pointcloud_file)
result      = predictor(data)
o3d.visualization.draw()
