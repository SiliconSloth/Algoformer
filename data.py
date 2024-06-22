import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import random


DataBatch = namedtuple("DataBatch", (
    "ops_a", "ops_b", "results",
    "digits_in", "mask", "mask_bounds"
))


def split_digits(value):
    digits = []
    while value > 0:
        digits.append(value % 10)
        value = value // 10
    return digits
    
def make_batch(n_digits, batch_size, max_tokens):
    ops_a, ops_b, results = [], [], []

    for _ in range(batch_size):
        a_len = random.randint(0, n_digits)
        b_len = random.randint(0, n_digits)

        a = random.randint(0, 10**a_len)
        b = random.randint(0, 10**b_len)
        c = a + b

        ops_a.append(a)
        ops_b.append(b)
        results.append(c)

    # Pack the sequences into tensors
    digits = np.zeros((batch_size, max_tokens), dtype=np.int64)
    mask = np.zeros((batch_size, max_tokens))
    mask_bounds = []

    for i, (a, b, c) in enumerate(zip(ops_a, ops_b, results)):
        row = split_digits(a) + [10] + split_digits(b) + [10]
        mask_start = len(row)
        row += split_digits(c) + [10]
        mask_end = len(row)
        row += [0] * (max_tokens - len(row))

        digits[i, :] = row
        mask[i, mask_start : mask_end] = 1
        mask_bounds.append((mask_start, mask_end))
    
    return DataBatch(
        ops_a, ops_b, results,
        digits, mask, mask_bounds
    )


# Generate training data in separate process to parallelise with training loop
class DataGenerator:
    def __init__(self, n_digits, batch_size, max_tokens) -> None:
        self.params = n_digits, batch_size, max_tokens

        self.executor = ProcessPoolExecutor(max_workers=1)
        self.future = None
    
    def next_batch(self):
        # Make first batch here, otherwise get previous result from executor
        batch = make_batch(*self.params) if self.future is None else self.future.result()
        # Start making next batch
        self.future = self.executor.submit(make_batch, *self.params)
        return batch
    
    def __del__(self):
        self.executor.shutdown()