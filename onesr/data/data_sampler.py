import math
import oneflow as flow
from oneflow.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from oneflow.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (flow.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = flow.Generator()
        g.manual_seed(self.epoch)
        indices = flow.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
