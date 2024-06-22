import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os, shutil, re

from data import DataGenerator
from model import Model


def prepare_directory():
    integer_pattern = re.compile("\\d+")

    root_path = "data/runs"
    os.makedirs(root_path, exist_ok=True)

    # Output directory names will start with a run number followed by an optional label (added manually).
    # Find the highest run number so far and use it to select this run's number.
    last_no = -1
    for file in os.listdir(root_path):
        if os.path.isdir(root_path + "/" + file):
            # Look for a run number at the start of the filename (match only checks the beginning of the string).
            run_no_match = integer_pattern.match(file)
            if run_no_match is not None:
                last_no = max(last_no, int(run_no_match.group()))
    run_no = last_no + 1
    run_path = root_path + "/" + str(run_no)
    print("Run number " + str(run_no))

    # Delete old output directories to save space. Labelled ones are ignored.
    for file in os.listdir(root_path):
        if file.isdigit() and int(file) <= run_no - 10:
            shutil.rmtree(root_path + "/" + file)

    # Record the code responsible for this run.
    script_path = run_path + "/scripts"
    os.makedirs(script_path, exist_ok=True)

    print("Copying scripts...")
    for file in os.listdir("."):
        if file.endswith(".py"):
            shutil.copy2(file, script_path)

    return run_path


if __name__ == "__main__":
    torch.set_default_device("cuda")
    run_path = prepare_directory()
    writer = SummaryWriter(run_path)

    n_digits = 6    # Maximum length of input numbers
    batch_size = 512
    max_tokens = 24 # Max sequence length
    vocab_size = 11 # Digits 0..9 + 10 to separate numbers                                                                                            

    data_generator = DataGenerator(n_digits, batch_size, max_tokens)
    model = Model(vocab_size)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    step = 0
    pass_count = 0
    while pass_count < 10:
        if step % 100 == 0:
            print(f"Step {step}")

        ops_a, ops_b, expected_results, digits_in, mask, mask_bounds = data_generator.next_batch()
        digits_in = torch.from_numpy(digits_in).cuda()
        mask = torch.from_numpy(mask).cuda()

        embeddings = nn.functional.one_hot(digits_in, vocab_size).to(torch.float).reshape(batch_size, max_tokens, vocab_size)
        embeddings = embeddings[:, :-1] # Drop last token for input sequence

        # Drop first token to align with model output, which excludes first token
        digits_in = digits_in[:, 1:]
        mask = mask[:, 1:]

        model.train()   # Ensure dropout is enabled for training
        output = model(embeddings) * mask[:, :, None] # Mask out non-output tokens to compute loss
        loss = criterion(output.reshape(batch_size * (max_tokens - 1), vocab_size), digits_in.flatten())

        if step % 100 == 0:
            writer.add_scalar("loss", loss, step)

            model.eval()    # Disable dropout to improve accuracy for evaluation
            eval_output = model(embeddings)
            digits_out = torch.argmax(eval_output, dim=-1)

            # Convert the output digits to numerical values
            values_out = []
            for digs, (m_start, m_end) in zip(digits_out.cpu(), mask_bounds):
                # Account for excluded first token in model output
                m_start -= 1
                m_end -= 2 # Also drop 10 from end of output
                
                # Extract and sum output digits
                res = digs[m_start : m_end].tolist()
                res = sum(r * 10**i for i, r in enumerate(res))
                values_out.append(res)

            # How many entire output sequences were correct
            accuracy = torch.mean((torch.tensor(values_out) == torch.tensor(expected_results)).to(float))
            writer.add_scalar("accuracy", accuracy, step)

            # How many individual output digits were correct
            digit_accuracy = torch.sum((digits_out == digits_in).to(float) * mask) / torch.sum(mask)
            writer.add_scalar("digit_accuracy", digit_accuracy, step)

            # Stop training once the accuracy is high enough for several consecutive steps
            if accuracy > 0.98:
                pass_count += 1
            else:
                pass_count = 0

            # Print the problems the model got wrong
            for op_a, op_b, true_out, pred_out in zip(ops_a, ops_b, expected_results, values_out):
                if pred_out != true_out:
                    print(f"{op_a} + {op_b} = {true_out} : {pred_out}")

            writer.flush()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    print("Done!")
    torch.save(model.state_dict(), run_path + "/model.pth")
    writer.close()