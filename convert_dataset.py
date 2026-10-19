"""
Convert Discord-Dialogues (one ChatML string per row) into a proper
`messages` chat format, shard it so every file is under the size cap,
and carve out a held-out eval split.

Input row  : text = "<|im_start|>user\nhey<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>"
Output row : messages = [{"role": "user", "content": "hey"},
                         {"role": "assistant", "content": "yo"}]

Usage (downloads from Hugging Face):
    python convert_dataset.py --out data

Usage (local parquet you already downloaded):
    python convert_dataset.py --input path/to/0000.parquet --out data
"""
import argparse
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

TURN_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)[ \t]*\n?(.*?)(?:<\|im_end\|>|(?=<\|im_start\|>)|\Z)",
    re.DOTALL,
)

MESSAGES_TYPE = pa.list_(
    pa.struct([("role", pa.string()), ("content", pa.string())])
)
SCHEMA = pa.schema([("messages", MESSAGES_TYPE), ("turns", pa.int32())])


def parse_chatml(text):
    """Split one ChatML string into a clean, strictly alternating message list.

    Returns None if the row can't make a valid user -> assistant conversation.
    """
    msgs = []
    for role, content in TURN_RE.findall(text):
        content = content.strip()
        if not content:
            continue
        if msgs and msgs[-1]["role"] == role:  # Gemma's template needs alternation
            msgs[-1]["content"] += "\n" + content
        else:
            msgs.append({"role": role, "content": content})

    # Must start with system/user, never with assistant.
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    # Must end on an assistant turn (that's what we train on).
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()

    body = [m for m in msgs if m["role"] != "system"]
    if len(body) < 2:
        return None
    return msgs


class ShardWriter:
    """Writes parquet shards, rotating before a file can reach the size cap."""

    def __init__(self, out_dir, prefix, max_bytes):
        self.out_dir, self.prefix, self.max_bytes = out_dir, prefix, max_bytes
        self.idx = -1
        self.writer = None
        self.path = None
        self.paths = []
        self.rows = 0

    def _open(self):
        self.idx += 1
        self.path = os.path.join(self.out_dir, f"{self.prefix}-{self.idx:05d}.parquet")
        self.writer = pq.ParquetWriter(self.path, SCHEMA, compression="zstd")
        self.paths.append(self.path)

    def write(self, table):
        if self.writer is None:
            self._open()
        self.writer.write_table(table)
        self.rows += table.num_rows
        # Row groups are flushed on write_table, so the on-disk size is current.
        if os.path.getsize(self.path) >= self.max_bytes:
            self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mookiezi/Discord-Dialogues",
                    help="HF dataset id (ignored if --input is given)")
    ap.add_argument("--input", help="local parquet file or folder instead of downloading")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-shard-mb", type=int, default=400,
                    help="rotate to a new file at this size; keep well under 500")
    ap.add_argument("--eval-every", type=int, default=1000,
                    help="every Nth kept row goes to the eval split (1000 = ~0.1%%)")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="stop after this many kept rows (0 = all)")
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--sample-rows", type=int, default=0,
                    help="randomly sample this many SOURCE rows first (0 = all); "
                         "deterministic for a given --seed")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    if args.max_shard_mb >= 500:
        sys.exit("--max-shard-mb must be below 500")
    os.makedirs(args.out, exist_ok=True)

    if args.input:
        files = args.input
        if os.path.isdir(files):
            files = [os.path.join(files, f) for f in sorted(os.listdir(files))
                     if f.endswith(".parquet")]
        ds = load_dataset("parquet", data_files=files, split="train")
    else:
        ds = load_dataset(args.dataset, split="train")
    print(f"loaded {len(ds):,} rows; columns = {ds.column_names}")
    if args.sample_rows and args.sample_rows < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.sample_rows))
        print(f"sampled {len(ds):,} rows (seed {args.seed})")

    max_bytes = args.max_shard_mb * 1024 * 1024
    train_w = ShardWriter(args.out, "train", max_bytes)
    eval_w = ShardWriter(args.out, "eval", max_bytes)

    kept = dropped = 0
    done = False
    for batch in ds.iter(batch_size=args.batch_size):
        tr, ev = [], []
        for text in batch[args.text_column]:
            msgs = parse_chatml(text)
            if msgs is None:
                dropped += 1
                continue
            kept += 1
            (ev if kept % args.eval_every == 0 else tr).append(msgs)
            if args.max_rows and kept >= args.max_rows:
                done = True
                break
        for rows, w in ((tr, train_w), (ev, eval_w)):
            if rows:
                w.write(pa.table(
                    {"messages": rows, "turns": [len(r) for r in rows]},
                    schema=SCHEMA,
                ))
        if kept and kept % 500_000 < args.batch_size:
            print(f"  kept {kept:,}  dropped {dropped:,}")
        if done:
            break
    train_w.close()
    eval_w.close()

    print(f"\nkept {kept:,} rows, dropped {dropped:,}")
    print(f"train rows {train_w.rows:,} | eval rows {eval_w.rows:,}")
    worst = 0
    for p in train_w.paths + eval_w.paths:
        mb = os.path.getsize(p) / 1024 / 1024
        worst = max(worst, mb)
        print(f"  {p}  {mb:.1f} MB")
    assert worst < 500, f"a shard is {worst:.1f} MB, over the 500 MB cap"
    print(f"all files under 500 MB (largest {worst:.1f} MB)")


if __name__ == "__main__":
    main()
