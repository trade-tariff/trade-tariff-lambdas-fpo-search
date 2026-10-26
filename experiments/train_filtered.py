"""Train the network from target/knn_index.pt with tradeset pairs filtered.

Uses the scores from experiments/semantic_filter_analysis.py. Only tradeset
rows are filtered. Writes model.pt, model.toml and running_losses.json to the
output directory.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import toml
import torch

from train_args import TrainScriptArgsParser
from training.train_model import FlatClassifierModelTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--rule", choices=["none", "chapter_rank_gt", "sim_lt"])
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--scores", default="benchmarking_data/experiments/002/pair_scores.npz"
)
cli, _ = parser.parse_known_args()

torch.manual_seed(cli.seed)
args = TrainScriptArgsParser()
args.load_config_file()
out = Path(cli.output_dir)
out.mkdir(parents=True, exist_ok=True)

subheadings = json.loads((args.target_dir() / "subheadings.json").read_text())
n_labels = len(subheadings)
index = torch.load(args.target_dir() / "knn_index.pt", weights_only=False, mmap=True)
is_tradeset = torch.tensor(
    ["tradesets_descriptions" in name for name in index["source_names"]]
)
row_texts = index["row_texts"].long()
row_labels = index["row_labels"].long()
row_tradeset = is_tradeset[index["row_sources"].long()]

keep = torch.ones(len(row_labels), dtype=torch.bool)
if cli.rule != "none":
    scores = np.load(cli.scores)
    pair_key = (
        scores["text"].astype(np.int64) * n_labels + scores["label"]
    ) * 2 + scores["tradeset"]
    if cli.rule == "chapter_rank_gt":
        remove = scores["chapter_rank"] > cli.threshold
    else:
        remove = scores["sim_own"] < cli.threshold
    remove &= scores["tradeset"]
    removed_keys = torch.from_numpy(np.sort(pair_key[remove]))
    row_key = (row_texts * n_labels + row_labels) * 2 + row_tradeset.long()
    pos = torch.searchsorted(removed_keys, row_key).clamp(max=len(removed_keys) - 1)
    keep = removed_keys[pos] != row_key

print(
    f"rule={cli.rule} threshold={cli.threshold} kept {int(keep.sum())} of "
    f"{len(keep)} rows; tradeset rows removed "
    f"{int((~keep).sum())} of {int(row_tradeset.sum())}"
)

device = args.torch_device()
embeddings = index["embeddings"].to(device)
state_dict, input_size, hidden_size, output_size = FlatClassifierModelTrainer(args).run(
    embeddings, row_labels[keep], n_labels, text_indexes=row_texts[keep]
)

torch.save(state_dict, out / "model.pt")
with open(out / "model.toml", "w") as f:
    toml.dump(
        {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "dropout_layer_1_percentage": args.model_dropout_layer_1_percentage(),
            "dropout_layer_2_percentage": args.model_dropout_layer_2_percentage(),
        },
        f,
    )
shutil.move("running_losses.json", out / "running_losses.json")
