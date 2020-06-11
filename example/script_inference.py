# import open3d as o3d
import torch
import open3d as o3d
import numpy as np

from ml3d import Predictor
from ml3d.util import Config


config_file     = 'ml3d/config/semantic_segmentation/randlanet_semantickitti.py'
checkpoint_file = 'ml3d/checkpoint/randlanet_semantickitti.pth'
pointcloud_file = 'datasets/fragment.ply'

def main():

    pcd = o3d.io.read_point_cloud(pointcloud_file)
    np_input    = np.concatenate((np.asarray(pcd.points), 
                        np.asarray(pcd.colors)), axis=1).astype(np.float32)



    cfg         = Config.load_from_file(config_file)

    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    predictor   = Predictor(cfg, checkpoint_file, device=device)

    data        = torch.from_numpy(np_input).unsqueeze(0).to(device)
    #data        = 1000*torch.randn(1, 2**16, 6).to(device)

    result      = predictor(data)
    predictions = torch.max(result, dim=-2).indices

    o3d.visualization.draw_geometries([pcd])
    print(predictions.size())
    #o3d.visualization.draw()
                
if __name__ == '__main__':
    main()
