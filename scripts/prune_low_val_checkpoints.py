"""
Run with:
    python -m scripts.prune_low_val_checkpoints.py
"""


import os

import torch


def main():
    print("Collecting files to delete")
    to_delete = []
    for log_dir in ("11-25", "11-27"): # TODO(piyush) Don't hardcode
        path = f"logs/{log_dir}"
        for exp_dir in os.listdir(path):
            exp_path = f"{path}/{exp_dir}"
            for model_dir in os.listdir(exp_path):
                model_path = f"{exp_path}/{model_dir}"
                if not os.path.isdir(model_path):
                    continue
                print(f"Processing {model_path}")

                checkpoints = {}
                for epoch_file in sorted(os.listdir(model_path)):
                    if epoch_file.endswith(".pt"):
                        try:
                            checkpoints[epoch_file] = torch.load(f"{model_path}/{epoch_file}", map_location="cpu")
                        except:
                            print(f"Loading failed for: {model_path}")

                best_val_acc = - float("inf")
                for epoch_file, checkpoint in checkpoints.items():
                    val_acc = checkpoint["val_acc"]
                    if val_acc < best_val_acc:
                        to_delete.append(f"{model_path}/{epoch_file}")
                    else:
                        best_val_acc = val_acc

    with open("to_delete.txt", "w") as f:
        f.write("\n".join(to_delete) + "\n")
    print("Dumped to to_delete.txt")

    # Delete.
    for epoch_path in to_delete:
        print(f"Deleting: {epoch_path}")
        os.remove(epoch_path)

if __name__ == "__main__":
    main()
