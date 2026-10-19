#!/usr/bin/env python3
"""
Run this on YOUR computer to watch a Salad training run and fetch the result.

    pip install boto3
    python salad_cli.py status
    python salad_cli.py download            # saves the finished LoRA to ./gemma4-discord-lora
    python salad_cli.py download --out my_folder

It reads R2_* settings from the environment or from a .env file in this folder.
"""
import argparse
import os
import sys
import time

import cloud


def load_dotenv(path=".env"):
    if not os.path.isfile(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0]  # tolerate an accidental trailing comment
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ago(ts):
    if not ts:
        return "never"
    s = int(time.time() - ts)
    return f"{s // 3600}h {s % 3600 // 60}m ago" if s >= 3600 else f"{s // 60}m {s % 60}s ago"


def status(store):
    st = store.get_json("state.json")
    if not st:
        print("No run found yet under this R2 prefix (the container hasn't started, or the prefix differs).")
        return
    print(f"status            : {st['status'].upper()}")
    print(f"container starts  : {st['boots']}   (more than 1 means the node was interrupted/restarted)")
    print(f"failures          : {st['failures']}")
    print(f"dataset ready     : {st['data_ready']}")
    print(f"latest checkpoint : {st['latest_checkpoint']} (step {st['latest_step']})")
    print(f"first started     : {ago(st['first_start'])}")
    print(f"last update       : {ago(st.get('updated_at'))}")
    if st.get("last_error"):
        print(f"last error        : {st['last_error'][:300]}")
        print("  full logs are in R2 under logs/failure-N.txt")
    if st["status"] == "done":
        print("\nFinished. Run:  python salad_cli.py download")


def download(store, out):
    st = store.get_json("state.json")
    if not st or st["status"] != "done":
        sys.exit("The run isn't finished yet (status is not DONE). Check with: python salad_cli.py status")
    n = store.download_dir("final-lora", out)
    print(f"Downloaded {n} files to {os.path.abspath(out)}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    d = sub.add_parser("download")
    d.add_argument("--out", default="gemma4-discord-lora")
    args = ap.parse_args()
    try:
        store = cloud.from_env()
    except RuntimeError as e:
        sys.exit(f"{e}\nFill in your .env file (copy .env.example) or set them in your shell.")
    status(store) if args.cmd == "status" else download(store, args.out)


if __name__ == "__main__":
    main()
