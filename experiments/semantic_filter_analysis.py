"""Score training pairs against CN self-texts to calibrate a semantic filter.

Reads target/knn_index.pt (built by `train.py --knn-index-only`), so no
training descriptions are embedded again. Writes per-pair scores and summary
tables to the output directory.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from train_args import TrainScriptArgsParser
from training.cleaning_pipeline import DescriptionLower, NegationCleaning

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="benchmarking_data/experiments/002")
parser.add_argument(
    "--benchmark-details",
    nargs="*",
    default=[
        "benchmarking_data/experiments/001_knn/flat/benchmark_details_good_goods.json",
        "benchmarking_data/experiments/001_knn/flat/benchmark_details_classified.json",
    ],
)
cli, _ = parser.parse_known_args()

args = TrainScriptArgsParser()
args.load_config_file()
device = "cuda"
out = Path(cli.output_dir)
out.mkdir(parents=True, exist_ok=True)

subheadings = json.loads((args.target_dir() / "subheadings.json").read_text())
code_to_label = {code: i for i, code in enumerate(subheadings)}

# Self-text reference embeddings, one per subheading label
negation = NegationCleaning.build()
lower = DescriptionLower()
self_texts: dict[str, str] = {}
with open(args.cn_data_file(), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        code = row["CN_CODE"].replace(" ", "")
        if len(code) == 8 and code.isdigit():
            _, text, _ = negation.filter(
                code, lower.filter(code, row["SelfText_EN"])[1]
            )
            self_texts[code] = text

ref_labels = [code_to_label[c] for c in self_texts if c in code_to_label]
ref_codes = [subheadings[i] for i in ref_labels]
model = SentenceTransformer(args.transformer(), device=device)
ref_emb = model.encode(
    [self_texts[c] for c in ref_codes],
    convert_to_tensor=True,
    normalize_embeddings=True,
    batch_size=256,
).half()

# Columns sorted by chapter so each chapter is a contiguous block
order = sorted(range(len(ref_codes)), key=lambda i: ref_codes[i])
ref_codes = [ref_codes[i] for i in order]
ref_labels = [ref_labels[i] for i in order]
ref_emb = ref_emb[order]
chapters = sorted({c[:2] for c in ref_codes})
chapter_bounds = []
for ch in chapters:
    idx = [i for i, c in enumerate(ref_codes) if c[:2] == ch]
    chapter_bounds.append((idx[0], idx[-1] + 1))
label_to_col = torch.full((len(subheadings),), -1, dtype=torch.long)
label_to_col[torch.tensor(ref_labels)] = torch.arange(len(ref_labels))
label_to_col = label_to_col.to(device)
col_chapter = torch.tensor(
    [chapters.index(c[:2]) for c in ref_codes], device=device, dtype=torch.long
)


def score(emb: torch.Tensor, labels: torch.Tensor, batch: int = 32768) -> dict:
    """Return own similarity, code rank, chapter rank and best chapter."""
    res = {k: [] for k in ("sim_own", "code_rank", "chapter_rank", "best_chapter")}
    for start in range(0, len(labels), batch):
        e = emb[start : start + batch].to(device, torch.float16)
        lab = labels[start : start + batch].to(device)
        cols = label_to_col[lab]
        sims = (e @ ref_emb.T).float()
        own = sims.gather(1, cols.clamp(min=0)[:, None]).squeeze(1)
        code_rank = (sims > own[:, None]).sum(1) + 1
        ch_max = torch.stack([sims[:, a:b].max(1).values for a, b in chapter_bounds], 1)
        own_ch = col_chapter[cols.clamp(min=0)]
        own_ch_max = ch_max.gather(1, own_ch[:, None]).squeeze(1)
        chapter_rank = (ch_max > own_ch_max[:, None]).sum(1) + 1
        valid = cols >= 0
        res["sim_own"].append(torch.where(valid, own, torch.nan).cpu())
        res["code_rank"].append(torch.where(valid, code_rank, -1).cpu())
        res["chapter_rank"].append(torch.where(valid, chapter_rank, -1).cpu())
        res["best_chapter"].append(ch_max.argmax(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in res.items()}


# Training pairs: unique (text, label) per source kind, weighted by row count
index = torch.load(args.target_dir() / "knn_index.pt", weights_only=False, mmap=True)
source_names = index["source_names"]
is_tradeset = torch.tensor(
    ["tradesets_descriptions" in name for name in source_names], dtype=torch.bool
)
row_texts = index["row_texts"].long()
row_labels = index["row_labels"].long()
row_tradeset = is_tradeset[index["row_sources"].long()]

n_labels = len(subheadings)
key = (row_texts * n_labels + row_labels) * 2 + row_tradeset.long()
keys, rows = torch.unique(key, return_counts=True)
pair_tradeset = (keys % 2).bool()
pair_label = (keys // 2) % n_labels
pair_text = (keys // 2) // n_labels

embeddings = index["embeddings"]
scores = score(embeddings[pair_text], pair_label)
np.savez_compressed(
    out / "pair_scores.npz",
    text=pair_text.numpy(),
    label=pair_label.numpy(),
    tradeset=pair_tradeset.numpy(),
    rows=rows.numpy(),
    **scores,
)


# Reference sets: benchmark pairs (trusted) and known incorrect pairs
def score_texts(pairs: list[tuple[str, str]]) -> dict:
    pairs = [(d, c) for d, c in pairs if c in code_to_label]
    emb = model.encode(
        [d for d, _ in pairs], convert_to_tensor=True, normalize_embeddings=True
    )
    return score(emb, torch.tensor([code_to_label[c] for _, c in pairs]))


bench_pairs = []
for path in cli.benchmark_details:
    bench_pairs += [
        (x["description"], x["correct_code"])
        for x in json.loads(Path(path).read_text())
    ]
bad_pairs = []
with open(args.incorrect_description_pairs_file(), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        bad_pairs.append(
            (row["GDSDESC"].lower().strip(), row["CMDTYCODE_Seen"].strip())
        )

ref_scores = {
    "benchmark": score_texts(bench_pairs),
    "known_bad": score_texts(bad_pairs),
}
np.savez_compressed(
    out / "reference_scores.npz",
    **{f"{n}_{k}": v for n, s in ref_scores.items() for k, v in s.items()},
)
print(
    f"pairs={len(keys)} tradeset_pairs={int(pair_tradeset.sum())} "
    f"benchmark={len(ref_scores['benchmark']['sim_own'])} "
    f"known_bad={len(ref_scores['known_bad']['sim_own'])}"
)
