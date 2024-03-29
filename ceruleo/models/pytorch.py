import torch
from ceruleo.iterators.iterators import WindowedDatasetIterator
import math 

class IterableDataset(torch.utils.data.IterableDataset):
     def __init__(self, iterator:WindowedDatasetIterator):
         super(IterableDataset).__init__()
         self.iterator = iterator

     def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
         return iter(range(iter_start, iter_end))