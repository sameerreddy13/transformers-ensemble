import argparse
import datetime
import itertools
import os
import pickle
import random

import torch
import transformers


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--save-dir",
        type=str,
        default="paramcounts_n32_labels2_hsize_numhlayers_nattheads_fcsize",
    )
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--save-freq", type=int, default=1000)
    ap.add_argument("--model", type=str, default="albert")

    return ap.parse_args()


def compute_vals(args):
    task_id, grid, vals = args
    filename = os.path.join(ARGS.save_dir, f"vals_{task_id}.pkl")

    for i, (
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
    ) in enumerate(grid):
        if ARGS.model == "albert":
            config = transformers.AlbertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_labels=2,
            )
            model = transformers.AlbertForSequenceClassification(config)
        elif ARGS.model == "bert":
            if hidden_size % num_attention_heads != 0:
                hidden_size = (hidden_size // num_attention_heads) * num_attention_heads
                if (
                    hidden_size,
                    num_hidden_layers,
                    num_attention_heads,
                    intermediate_size,
                ) in vals:
                    continue
            config = transformers.BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
            )
            model = transformers.BertForSequenceClassification(config)
        else:
            raise ValueError(f"Unknown model {ARGS.model}")

        n_params = sum([param.numel() for param in model.parameters()])
        vals[(hidden_size, num_hidden_layers, num_attention_heads, intermediate_size)] = n_params

        if i % ARGS.save_freq == 0:
            with open(filename, "wb") as f:
                pickle.dump(vals, f)
            print(
                f"[Process {task_id}] [{datetime.datetime.now()}] [Iter {i}] Saved progress to",
                filename,
            )

    with open(filename, "wb") as f:
        pickle.dump(vals, f)
    print(f"[Process {task_id}] Done and saved to {filename}")


def main():
    os.makedirs(ARGS.save_dir, exist_ok=True)
    print(f"Saving outputs to {ARGS.save_dir}")

    if ARGS.model == "albert":
        grid = list(
            itertools.product(
                *[
                    [int(4096 * k / ARGS.n) for k in range(1, ARGS.n + 1)],  # hidden_size
                    range(1, 12 + 1),  # num_hidden_layers
                    range(1, 64 + 1),  # num_attention_heads
                    [int(16384 * k / ARGS.n) for k in range(1, ARGS.n + 1)],  # intermediate_size
                ]
            )
        )
    elif ARGS.model == "bert":
        grid = list(
            itertools.product(
                *[
                    [int(768 * k / ARGS.n) for k in range(1, ARGS.n + 1)],  # hidden_size
                    range(1, 12 + 1),  # num_hidden_layers
                    range(1, 12 + 1),  # num_attention_heads
                    [int(3072 * k / ARGS.n) for k in range(1, ARGS.n + 1)],  # intermediate_size
                ]
            )
        )
    else:
        raise ValueError(f"Unknown model {ARGS.model}")
    random.shuffle(grid)
    print(f"Created {len(grid)} grid search points")

    vals = {}
    if ARGS.resume:
        for filename in os.listdir(ARGS.save_dir):
            with open(os.path.join(ARGS.save_dir, filename), "rb") as f:
                vals.update(pickle.load(f))

        grid = [tup for tup in grid if tup not in vals]
        print(f"Resuming from {ARGS.save_dir}/ - {len(grid)} points remaining")

    chunk_size = len(grid) // ARGS.num_processes
    chunks = [
        (i, grid[i * chunk_size : (i + 1) * chunk_size], vals if i == 0 else {})
        for i in range(ARGS.num_processes)
    ]

    print(f"Launching with {ARGS.num_processes} processes")
    pool = torch.multiprocessing.Pool(ARGS.num_processes)
    pool.map(compute_vals, chunks)


if __name__ == "__main__":
    ARGS = parse_args()
    main()
