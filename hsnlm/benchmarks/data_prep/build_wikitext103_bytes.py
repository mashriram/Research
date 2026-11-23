"""Builds data/wikitext103_raw.bytes for gpu_wikitext103_3seed.py: first 40,000,000 UTF-8 bytes of the WikiText-103
raw (v1, untokenized) TRAIN split, immediately followed by the first 1,000,000 bytes of the official VALIDATION split,
so ByteData(path, n_train=40_000_000, n_val=1_000_000) reads train/val with the same sizes as the enwik8 runs.
Source: HF hub Salesforce/wikitext, config wikitext-103-raw-v1 (rows are lines incl. trailing newline; joined as-is)."""
import sys, hashlib, pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]
def text(f): return "".join(pq.read_table(f"{src}/{f}", columns=["text"]).column("text").to_pylist()).encode("utf-8")
tr, va = text("train-00000-of-00002.parquet"), text("validation-00000-of-00001.parquet")
assert len(tr) >= 40_000_000 and len(va) >= 1_000_000, (len(tr), len(va))
blob = tr[:40_000_000] + va[:1_000_000]
open(out, "wb").write(blob)
print("train shard bytes", len(tr), "val bytes", len(va), "wrote", len(blob), "sha256", hashlib.sha256(blob).hexdigest())
print("train head:", repr(tr[:200].decode("utf-8", "replace"))); print("val head:", repr(va[:200].decode("utf-8", "replace")))
