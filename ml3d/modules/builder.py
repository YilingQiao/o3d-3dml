from ml3d.util import Registry

NETWORK = Registry('network')

def build(cfg, registry, args=None)
    return build_from_cfg(cfg, registry, args)

def build_network(cfg)
    return build(cfg, NETWORK)