import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os, shutil, re
import argparse

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

    # Record the code responsible for this run.
    script_path = run_path + "/scripts"
    os.makedirs(script_path, exist_ok=True)

    print("Copying scripts...")
    for file in os.listdir("."):
        if file.endswith(".py"):
            shutil.copy2(file, script_path)

    return run_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-blanks", action="store_true", help="Disable blank intermediate tokens")
    parser.add_argument("--pos-offset", action="store_true", help="Offset position encodings to emulate presence of intermediate tokens")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    run_path = prepare_directory()
    writer = SummaryWriter(run_path)

    n_digits = 48 # Maximum list length
    batch_size = 342
    vocab_size = 12 # Digits 0..9 + 10 to terminate list and 11 for blank tokens

    data_generator = DataGenerator(n_digits, batch_size, 5,
                                   use_intermediates=not args.no_blanks)
    output_offset = (n_digits + 1) if args.pos_offset else 0
    model = Model(vocab_size, output_offset=output_offset)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    max_step = 6000
    for step in range(max_step):
        if step % 10 == 0:
            print(f"Step {step}")

        samples, digits_in, digits_out, attn_mask, output_mask, mask_bounds = data_generator.next_batch()
        digits_in = torch.from_numpy(digits_in).cuda()
        digits_out = torch.from_numpy(digits_out).cuda()
        attn_mask = torch.from_numpy(attn_mask).cuda()
        output_mask = torch.from_numpy(output_mask).cuda()
        
        batch_size = digits_in.shape[0]
        inp_len = digits_in.shape[1]
        out_len = digits_out.shape[1]

        emb_in = nn.functional.one_hot(digits_in, vocab_size).to(torch.float).reshape(batch_size, inp_len, vocab_size)
        emb_out = nn.functional.one_hot(digits_out, vocab_size).to(torch.float).reshape(batch_size, out_len, vocab_size)

        sequence = torch.cat([emb_in, emb_out[:, :-1]], dim=1)

        model.train()   # Ensure dropout is enabled for training
        output = model(sequence, attn_mask)[:, -out_len:] * output_mask[:, :, None]
        loss = criterion(output.reshape(batch_size * out_len, vocab_size), digits_out.flatten())

        if step % 10 == 0:
            writer.add_scalar("loss", loss, step)

            model.eval()    # Disable dropout to improve accuracy for evaluation
            eval_output = model(sequence, attn_mask)[:, -out_len:]
            preds_out = torch.argmax(eval_output, dim=-1)

            # How many entire output sequences were correct
            errors = torch.sum((preds_out != digits_out).to(float) * output_mask, dim=1)
            accuracy = 1 - (torch.count_nonzero(errors) / batch_size)
            writer.add_scalar("accuracy", accuracy, step)

            # How many individual output digits were correct
            digit_accuracy = 1 - (torch.sum(errors) / torch.sum(output_mask))
            writer.add_scalar("digit_accuracy", digit_accuracy, step)

            writer.flush()
        
        if step % 500 == 0 and step > 0:
            torch.save(model.state_dict(), run_path + f"/model_{step}.pth")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Done!")
    torch.save(model.state_dict(), run_path + f"/model_{max_step}.pth")
    writer.close()