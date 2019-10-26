
import itertools

from torch.utils import data

class MergedDataset(data.Dataset):
    """
    Merges datasets that each return tuples of results.
    Each individual dataset should return a tuple.
    This class will return a concatenated tuple from each concatenated member.
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        assert len(set([len(d) for d in datasets])) == 1

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        return tuple(itertools.chain(*[d[index] for d in self.datasets]))

