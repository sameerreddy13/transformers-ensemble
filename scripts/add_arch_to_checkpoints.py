"""
Run with:
    python -m scripts.add_arch_to_checkpoints
"""

import os

import torch

import utils


def main():
    for log_dir in ("11-25", "11-27"): # TODO(piyush) Don't hardcode
        path = f"logs/{log_dir}"

        for exp_dir in os.listdir(path):
            exp_path = f"{path}/{exp_dir}"

            model_paths = [f"{exp_path}/{model_dir}" for model_dir in os.listdir(exp_path)]
            model_paths = [path for path in model_paths if os.path.isdir(path)]

            if log_dir == "11-25":
                if "subnet" in exp_dir:
                    arch = {"num_hidden_layers": 6, "num_attention_heads": 6,
                            "intermediate_size": 3072 // 2}
                elif "naive" in exp_dir:
                    arch = {"num_hidden_layers": 3, "intermediate_size": 3072 // 4}
                else:
                    raise ValueError("Unknown experiment")
            elif log_dir == "11-27":
                arch = utils.get_subnet_configs_fixed(num_models=len(model_paths))[0]
            else:
                raise ValueError("This should not happen")

            for model_path in model_paths:
                for epoch_file in sorted(os.listdir(model_path)):
                    if not epoch_file.endswith(".pt"):
                        continue
                    ckpt_path = f"{model_path}/{epoch_file}"

                    try:
                        checkpoint = torch.load(ckpt_path, map_location="cpu")
                    except:
                        print(f"Loading failed for: {ckpt_path}")
                        continue

                    checkpoint["arch"] = arch
                    torch.save(checkpoint, ckpt_path)
                    print(f"Updated {ckpt_path} with: {arch}")



if __name__ == "__main__":
    main()
