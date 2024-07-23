import numpy as np
from multiprocessing import Process, Queue
from collections import namedtuple
from itertools import chain
import random


DataBatch = namedtuple("DataBatch", (
    "samples", "digits_in", "mask", "mask_bounds"
))


def make_sample(n_digits):
    seq_len = random.randint(1, n_digits)
    inp = [random.randint(0, 9) for _ in range(seq_len)]

    stages = []
    cur = list(inp)
    already_sorted = False
    while not already_sorted:
        already_sorted = True
        # Run an iteration of bubble sort
        for i in range(len(cur) - 1):
            if cur[i] > cur[i + 1]:
                tmp = cur[i]
                cur[i] = cur[i + 1]
                cur[i + 1] = tmp
                already_sorted = False
        
        # Don't add the stage to output if it was already sorted,
        # unless the input list was already sorted so this is the only stage.
        if not (already_sorted and stages):
            stages.append(cur + [10])
    
    # Output stage does not have 10, and is surrounded by 11
    stages[-1] = [11] + stages[-1][:-1] + [11]
    out = list(chain(*stages))

    inp.append(10)
    return inp, out


class BatchDispatcher:
    def __init__(self, batch_size, max_tokens, queue) -> None:
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.queue = queue

        self.samples = []
    
    def put_sample(self, sample):
        # Once the batch size is reached, evict an existing sample
        # at random each time a new one is added.
        if len(self.samples) < self.batch_size:
            self.samples.append(sample)
        else:
            i = random.randint(0, len(self.samples) - 1)
            self.samples[i] = sample

    def pack_batch(self):
        digits = np.zeros((len(self.samples), self.max_tokens), dtype=np.int64)
        mask = np.zeros((len(self.samples), self.max_tokens))
        mask_bounds = []

        for i, (inp, out) in enumerate(self.samples):
            mask_start = len(inp)
            row = inp + out
            mask_end = len(row)
            row += [0] * (self.max_tokens - len(row))

            mask[i, mask_start : mask_end] = 1

            digits[i, :] = row
            mask_bounds.append((mask_start, mask_end))
        
        return DataBatch(
            self.samples, digits, mask, mask_bounds
        )
    
    def try_dispatch(self):
        if len(self.samples) == self.batch_size:
            self.queue.put(self.pack_batch())


# Generate training data in separate process to parallelise with training loop
class DataGenerator:
    def __init__(self, n_digits, total_batch_elems, bucket_width) -> None:
        self.n_digits = n_digits
        self.total_batch_elems = total_batch_elems
        self.bucket_width = bucket_width

        self.queue = Queue(maxsize=1)
        self.dispatchers = []

        process = Process(target=self.generator_loop, daemon=True)
        process.start()
    
    def get_dispatcher(self, bucket):
        while len(self.dispatchers) <= bucket:
            max_tokens = (bucket + 1) * self.bucket_width
            batch_size = self.total_batch_elems // max_tokens
            self.dispatchers.append(BatchDispatcher(batch_size, max_tokens, self.queue))
        return self.dispatchers[bucket]

    def generator_loop(self):
        while True:
            sample = make_sample(self.n_digits)
            sample_len = len(sample[0]) + len(sample[1])
            bucket = sample_len // self.bucket_width
            self.get_dispatcher(bucket).put_sample(sample)

            if not self.queue.full():
                random.choice(self.dispatchers).try_dispatch()
    
    def next_batch(self):
        return self.queue.get()