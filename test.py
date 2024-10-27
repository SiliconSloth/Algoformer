import numpy as np
import torch
from torch import nn
import argparse
import random

from model import Model


parser = argparse.ArgumentParser()
parser.add_argument("--mask-blanks", action="store_true", help="Mask out blank tokens")
args = parser.parse_args()

torch.set_default_device("cuda")

max_digits = 48
n_runs = 200
vocab_size = 12
block_length = max_digits + 1

model_path = "data/runs/0/model_3500.pth"

model = Model(vocab_size)
model.load_state_dict(torch.load(model_path))
model.eval()

results = []
for run_no in range(1, n_runs + 1):
    unsorted = np.random.randint(10, size=random.randint(1, max_digits)).tolist()
    print(f"Input  {run_no}/{n_runs}:")
    print(unsorted)

    digits_in = torch.tensor(unsorted + [10])
    pad_len = block_length - digits_in.shape[0]

    embeddings = nn.functional.one_hot(digits_in, vocab_size).to(torch.float).unsqueeze(0)
    embeddings = torch.concat([torch.zeros([1, pad_len, vocab_size]), embeddings], dim=1)

    mask = nn.Transformer.generate_square_subsequent_mask(block_length * 3, device="cuda")
    mask[pad_len:, :pad_len] = float("-inf")
    if args.mask_blanks:
        mask[block_length * 2:, block_length : block_length * 2] = float("-inf")

    result = []
    with torch.no_grad():
        for i in range(block_length * 2):
            # Generate the next digit based on the sequence so far
            digit_emb = model(embeddings, mask[:embeddings.shape[1], :embeddings.shape[1]])[:, -1:]
            digit = torch.argmax(digit_emb, dim=-1).item()
            if i >= block_length and len(result) < len(unsorted):
                result.append(digit)
                if len(result) == len(unsorted):
                    break
            
            # Append the generated digit to the end of the sequence,
            # ready to generate the next digit
            digit_emb[:] = 0
            digit_emb[:,:, digit] = 1
            embeddings = torch.concat([embeddings, digit_emb], dim=1)

    print(f"Output {run_no}/{n_runs}:")
    print(result)
    print()

    is_correct = result == sorted(unsorted)
    results.append(is_correct)

print(f"Accuracy: {sum(results) * 100 / len(results):.1f}%")
