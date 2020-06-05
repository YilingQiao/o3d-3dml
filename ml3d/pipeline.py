import warnings
import torch

from ml3d.module import build_network
from ml3d.module.misc import Composer
from ml3d.util import load_checkpoint


class Predictor():
    """
    Predictor.
    """
    def __init__(self, cfg, checkpoint=None, device='cuda:0'):
        self.cfg     = cfg
        self.network = build_network(self.cfg.cfg_dict['network'])
        if checkpoint is not None:
            if device is 'cpu':
                checkpoint = load_checkpoint(self.network, checkpoint, torch.device(device))
            else:
                checkpoint = load_checkpoint(self.network, checkpoint, device)
        else :
            warnings.warn('No checkpoint file')

        self.network.to(device)
        pipeline = [self.network]
        self._pipeline = Composer(pipeline)

    def __call__(self, input_data):
        # print(input_data)
        # print(self._pipeline)
        with torch.no_grad():
            #result = self.network(input_data)
            result = self._pipeline(input_data)

        return result

