# datamaker.py
# Streams C4 from Hugging Face and writes ~100k-line chunk files with resume support.
# Uses split="train"/"validation" (no percent slice). Bound data via --fraction or --num_shards/--shard_index.

import argparse, json, os, sys, time
from typing import Dict, Any
from datasets import load_dataset

DEFAULT_OUT_DIR = "Data_Files"
BOOKMARK_NAME = "progress.json"
HUB_REPO = "allenai/c4"  # explicit hub path to avoid local c4.py collisions

def save_state(out_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, BOOKMARK_NAME), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def load_state(out_dir: str) -> Dict[str, Any] | None:
    p = os.path.join(out_dir, BOOKMARK_NAME)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def parse_args():
    ap = argparse.ArgumentParser(description="Stream C4 and dump in fixed-size line chunks.")
    ap.add_argument("--split", default="train", choices=["train", "validation"], help="Dataset split")
    ap.add_argument("--lang", default="en", help='C4 config (usually "en")')
    ap.add_argument("--target_lines", type=int, default=5_000_000, help="Total lines to write across all files")
    ap.add_argument("--chunk_lines", type=int, default=100_000, help="Lines per output file")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for streaming buffer shuffle")
    ap.add_argument("--shuffle_buffer", type=int, default=10_000, help="Streaming shuffle buffer size")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--no_shuffle", action="store_true", help="Disable streaming shuffle")
    ap.add_argument("--text_field", default="text", help='Field containing text (default "text")')
    # Bounding options (pick one):
    ap.add_argument("--fraction", type=float, default=None,
                    help="Approx fraction of data to read (e.g., 0.02 ≈ 2%) via sharding")
    ap.add_argument("--num_shards", type=int, default=None,
                    help="Divide stream into N shards (mutually exclusive with --fraction)")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="Shard index to read (0-based). Use with --num_shards")
    return ap.parse_args()

def apply_bounding(ds, args):
    # If user gave --fraction, convert to an integer shard scheme
    if args.fraction is not None:
        if not (0 < args.fraction <= 1):
            raise ValueError("--fraction must be in (0, 1].")
        # choose num_shards so that 1/num_shards ≈ fraction
        num_shards = max(1, int(round(1.0 / args.fraction)))
        shard_index = 0  # take the first fraction; change if you want a different slice
        ds = ds.shard(num_shards=num_shards, index=shard_index)
        print(f"→ Applied fraction {args.fraction:.6f} via shard {shard_index}/{num_shards}")
        return ds

    # Else, if user gave explicit shards, apply those
    if args.num_shards is not None:
        if args.num_shards < 1:
            raise ValueError("--num_shards must be >= 1.")
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError("--shard_index must be in [0, num_shards).")
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)
        print(f"→ Applied shard index {args.shard_index} of {args.num_shards}")
        return ds

    # No bounding by shards; rely on --target_lines cap only
    return ds

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load or init bookmark
    state = load_state(args.out_dir) or {
        "hub_repo": HUB_REPO,
        "split": args.split,
        "lang": args.lang,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "target_lines": args.target_lines,
        "chunk_lines": args.chunk_lines,
        "total_lines_written": 0,
        "next_file_index": 1,
    }

    # Guard against param drift across resumes
    for key, cur in [("hub_repo", HUB_REPO), ("split", args.split), ("lang", args.lang)]:
        if state.get(key) != cur:
            print(
                f"[ERROR] Bookmark is for {key}={state.get(key)!r}, but you passed {cur!r}.\n"
                f"Delete {os.path.join(args.out_dir, BOOKMARK_NAME)} to start fresh, "
                f"or match your arguments to the bookmark.",
                file=sys.stderr,
            )
            sys.exit(1)

    written_total = int(state.get("total_lines_written", 0))
    file_idx = int(state.get("next_file_index", 1))

    print(f"→ Loading {HUB_REPO}/{args.lang} split={args.split} (streaming=True)")
    ds = load_dataset(HUB_REPO, args.lang, split=args.split, streaming=True)

    # Optional: shuffle with bounded buffer to keep memory small
    if not args.no_shuffle:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    # Apply bounding (fraction or shards) *before* skipping and writing
    ds = apply_bounding(ds, args)

    # Resume: skip what we've already written
    if written_total > 0:
        print(f"→ Resuming: skipping {written_total:,} examples already written…")
        ds = ds.skip(written_total)

    it = iter(ds)
    start = time.time()

    try:
        while written_total < args.target_lines:
            path = os.path.join(args.out_dir, f"data_{file_idx:05d}.txt")
            lines_in_chunk = 0
            with open(path, "w", encoding="utf-8") as f:
                while lines_in_chunk < args.chunk_lines and written_total < args.target_lines:
                    try:
                        ex = next(it)
                    except StopIteration:
                        print("Stream ended (slice exhausted).")
                        break

                    text = ex.get(args.text_field, "")
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    f.write(text.rstrip("\n") + "\n")

                    lines_in_chunk += 1
                    written_total += 1

                    if written_total % 50_000 == 0:
                        rate = written_total / max(1e-6, (time.time() - start))
                        print(f"… {written_total:,} lines written ({rate:,.1f} lines/s)")
                        state.update(total_lines_written=written_total, next_file_index=file_idx)
                        save_state(args.out_dir, state)

            if lines_in_chunk == 0:
                if written_total < args.target_lines:
                    print("No more data available in the requested slice; stopping.")
                break

            print(f"✓ Wrote {lines_in_chunk:,} lines → {path}")
            file_idx += 1
            state.update(total_lines_written=written_total, next_file_index=file_idx)
            save_state(args.out_dir, state)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Bookmark saved; rerun to resume.")
        state.update(total_lines_written=written_total, next_file_index=file_idx)
        save_state(args.out_dir, state)

    elapsed = time.time() - start
    rate = written_total / max(1e-6, elapsed)
    print(f"\nDone. Total lines: {written_total:,} in {elapsed:,.1f}s ({rate:,.1f} lines/s)")

if __name__ == "__main__":
    main()
