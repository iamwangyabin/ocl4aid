from torch.utils.data import Dataset

from .safe_sample import make_bad_sample


class OnlineIterDataset(Dataset):
    def __init__(self,
                 dataset   : Dataset,
                 ) -> None:
        super().__init__()
        self.dataset = dataset
        self.classes = dataset.classes
        self.targets = dataset.targets

    def __getitem__(self, index):
        try:
            image, label = self.dataset.__getitem__(index)
        except Exception as exc:
            return make_bad_sample(index, exc)
        return image, label, index

    def __len__(self):
        return len(self.dataset)
