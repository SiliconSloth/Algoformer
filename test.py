import torch
from torch import nn

from data import split_digits
from model import Model


torch.set_default_device("cuda")

# The input numbers, up to 6 digits long
operand_a = 584999
operand_b =  70768

model_path = "data/runs/0/model.pth"

max_tokens = 100 # Just in case the model tries to go on forever
vocab_size = 11

model = Model(vocab_size)
model.load_state_dict(torch.load(model_path))
model.eval()

digits_a = split_digits(operand_a)
digits_b = split_digits(operand_b)

digits_in = torch.tensor(digits_a + [10] + digits_b + [10])
embeddings = nn.functional.one_hot(digits_in, vocab_size).to(torch.float).unsqueeze(0)

result = 0
for i in range(max_tokens):
    # Generate the next digit based on the sequence so far
    output = model(embeddings)[:, -1:]
    digit_out = torch.argmax(output, dim=-1)

    # Stop once 10 is reached, as it marks the end of the output
    d = digit_out.item()
    if d == 10:
        break
    else:
        result += d * 10**i

    # Append the generated digit to the end of the sequence,
    # ready to generate the next digit
    embedding_out = nn.functional.one_hot(digit_out, vocab_size).to(torch.float)
    embeddings = torch.cat([embeddings, embedding_out], dim=1)

print(f"{operand_a} + {operand_b} = {result}")