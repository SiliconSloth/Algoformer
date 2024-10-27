import numpy as np
from multiprocessing import Process, Queue
import random


def make_sample(n_digits, use_intermediates):
    seq_len = random.randint(1, n_digits)
    inp = [random.randint(0, 9) for _ in range(seq_len)]
    
    out = sorted(inp)
    if use_intermediates:
        out = [11] * (n_digits + 1) + out
    inp.append(10)
    return inp, out


# Generate training data in separate process to parallelise with training loop
class DataGenerator:
    def __init__(self, n_digits, batch_size, n_heads, use_intermediates) -> None:
        self.n_digits = n_digits
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.use_intermediates = use_intermediates

        self.queue = Queue(maxsize=1)
        process = Process(target=self.generator_loop, daemon=True)
        process.start()

    def make_batch(self):
        samples = [make_sample(self.n_digits, self.use_intermediates) for _ in range(self.batch_size)]
        block_len = self.n_digits + 1
        output_len = block_len * 2 if self.use_intermediates else block_len

        digits_in = np.zeros((len(samples), block_len), dtype=np.int64)
        digits_out = np.zeros((len(samples), output_len), dtype=np.int64)
        output_mask = np.zeros((len(samples), output_len))
        mask_bounds = []

        mask_size = block_len + output_len - 1
        attn_mask = np.triu(np.ones((self.batch_size, mask_size, mask_size)), k=1).astype(bool)

        for i, (inp, out) in enumerate(samples):
            # Add leading padding tokens so all inputs end at block_len
            inp_start = block_len - len(inp)
            row = [0] * inp_start
            row += inp
            digits_in[i, :] = row

            row = out + [0] * (output_len - len(out))
            digits_out[i, :] = row

            mask_end = len(out)
            output_mask[i, : mask_end] = 1

            # Exclude leading padding from attention.
            # Don't mask attention for the padding tokens themselves,
            # to avoid NaNs during gradient descent.
            attn_mask[i, inp_start:, :inp_start] = True

            mask_bounds.append((0, mask_end))
        
        attn_mask = np.repeat(attn_mask, self.n_heads, axis=0)
        
        return (
            list(samples), digits_in, digits_out, attn_mask, output_mask, mask_bounds
        )

    def generator_loop(self):
        while True:
            self.queue.put(self.make_batch())
    
    def next_batch(self):
        return self.queue.get()