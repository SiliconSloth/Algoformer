import torch
from torch import nn

from model import Model


torch.set_default_device("cuda")

# The input list, up to 20 digits long
unsorted = [9, 8, 7, 6, 3, 1, 0, 2, 7]
print(unsorted)

model_path = "data/runs/0/model_10000.pth"

vocab_size = 12

model = Model(vocab_size)
model.load_state_dict(torch.load(model_path))
model.eval()

digits_in = torch.tensor(unsorted + [10])
embeddings = nn.functional.one_hot(digits_in, vocab_size).to(torch.float).unsqueeze(0)

result = []
outputting = False
while True:
    # Generate the next digit based on the sequence so far
    output = model(embeddings)[:, -1:]
    digit_out = torch.argmax(output, dim=-1)

    # Output all digits between the 11s then stop
    d = digit_out.item()
    if d == 11:
        if outputting:
            break
        else:
            outputting = True
    elif outputting:
        result.append(d)

    # Append the generated digit to the end of the sequence,
    # ready to generate the next digit
    embedding_out = nn.functional.one_hot(digit_out, vocab_size).to(torch.float)
    embeddings = torch.cat([embeddings, embedding_out], dim=1)

print(result)