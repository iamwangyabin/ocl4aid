from torch.utils.data import Dataset


class OnlineIterDataset(Dataset):
    def __init__(self,
                 dataset   : Dataset,
                 ) -> None:
        super().__init__()
        self.dataset = dataset
        self.classes = dataset.classes
        self.targets = dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image, label, index

    def __len__(self):
        return len(self.dataset)
