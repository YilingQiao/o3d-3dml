import warnings

from ml3d.module import build_network



class Predictor():
    """
    Predictor.
    """
    def __init__(self, cfg, checkpoint=None, device='cuda:0'):
        self.cfg     = cfg.clone()
        self.network = build_network(self.cfg.cfg_dic)
        if checkpoint is not None:
            checkpoint = load_checkpoint(self.network, checkpoint)
        else :
            warnings:.warn('No checkpoint file')

        pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]

        self._pipeline = Compose(pipeline)

    def __call__(self, input_data):

