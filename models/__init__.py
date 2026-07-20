from models.resnet import Resnet

class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(cfg):
        if cfg.get('model') == 'resnet':
            version = cfg.get('version', '18')
            num_classes = cfg.get('num_classes', 10)
            return Resnet(version, num_classes)
        else:
            raise NotImplementedError
