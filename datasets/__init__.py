from datasets.mnist_dataset import MnistDataset

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset_type(dataset_cfg: dict):
        if dataset_cfg.get('type') == 'mnist':
            return MnistDataset
        else:
            raise NotImplementedError