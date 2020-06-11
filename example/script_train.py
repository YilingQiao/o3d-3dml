# import open3d as o3d
import torch
import open3d as o3d
import numpy as np

from ml3d import Trainer
from ml3d.util import Config


config_file     = 'ml3d/config/semantic_segmentation/randlanet_semantickitti.py'
checkpoint_file = 'ml3d/checkpoint/randlanet_semantickitti.pth'
work_dir        = 'runs'
datasets_path   = 'datasets/s3dis'

def main():
    cfg             = Config.load_from_file(config_file)
    cfg.work_dir    = work_dir

    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer         = Trainer(cfg, datasets_path)
    trainer.train()

if __name__ == '__main__':
    main()
