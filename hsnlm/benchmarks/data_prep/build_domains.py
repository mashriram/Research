"""B3 data: builds data/domain_<name>.bytes = 40,000,000 train bytes + 1,000,000 held-out val bytes for six non-enwik8
domains, so every domain runs through the identical byte-level (vocab 256) pipeline as enwik8 / WikiText-103.
Downloads go to data/raw_domains/<name>/ ; nothing downloaded is executed or unpickled (CIFAR uses the raw *binary* release,
audio is parsed with the stdlib `wave` module from inside the zip).
  code   codeparrot/codeparrot-clean-valid (Python source)          train/val = disjoint files
  hindi  wikimedia/wikipedia 20231101.hi shard 1 (Devanagari UTF-8)  train/val = disjoint articles
  zh     wikimedia/wikipedia 20231101.zh shard 2 (Chinese UTF-8)      train/val = disjoint articles
  dna    UCSC hg38 chr20 (A/C/G/T only, N removed, uppercased)        train = first 40M bases, val = the next 1M
  image  CIFAR-10 binary: pixels in raster order, R,G,B interleaved   train = data_batch_1..5, val = test_batch (bits/dim)
  audio  TF mini Speech Commands, 16 kHz, 8-bit mu-law (WaveNet)     train/val = disjoint clips (val = last clips in sorted order)
Usage: python -I build_domains.py <hsnlm_root> [names...]"""
import sys, os, io, gzip, json, tarfile, zipfile, wave, hashlib, urllib.request
import numpy as np
ROOT = sys.argv[1]; NAMES = sys.argv[2:] or ["code", "hindi", "zh", "dna", "image", "audio"]
RAW = os.path.join(ROOT, "data", "raw_domains"); NT, NV = 40_000_000, 1_000_000

def fetch(url, name, fn):
    d = os.path.join(RAW, name); os.makedirs(d, exist_ok=True); p = os.path.join(d, fn)
    if not os.path.exists(p):
        print("download", url, flush=True); urllib.request.urlretrieve(url, p + ".part"); os.replace(p + ".part", p)
    return p
def hf(repo, path, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", local_dir=os.path.join(RAW, name))
def write(name, train, val):
    assert len(train) >= NT and len(val) >= NV, (name, len(train), len(val))
    blob = bytes(train[:NT]) + bytes(val[:NV]); out = os.path.join(ROOT, "data", f"domain_{name}.bytes")
    open(out, "wb").write(blob)
    print(name, "wrote", len(blob), "sha256", hashlib.sha256(blob).hexdigest()[:16], "| train head:", repr(bytes(train[:60])), flush=True)

def docs_split(docs, sep):  # accumulate docs until train full, then val from the following (disjoint) docs
    tr, va = bytearray(), bytearray()
    for d in docs:
        b = d.encode("utf-8") + sep
        if len(tr) < NT: tr += b
        elif len(va) < NV: va += b
        else: break
    return tr, va

if "code" in NAMES:
    p = hf("codeparrot/codeparrot-clean-valid", "file-000000000054.json.gz", "code")
    def it():
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f: yield json.loads(line)["content"]
    write("code", *docs_split(it(), b"\n"))
for name, cfg, shard in (("hindi", "20231101.hi", "train-00001-of-00002.parquet"), ("zh", "20231101.zh", "train-00002-of-00006.parquet")):
    if name in NAMES:
        import pyarrow.parquet as pq
        p = hf("wikimedia/wikipedia", f"{cfg}/{shard}", name)
        write(name, *docs_split(pq.read_table(p, columns=["text"]).column("text").to_pylist(), b"\n\n"))
if "dna" in NAMES:
    p = fetch("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz", "dna", "chr20.fa.gz")
    with gzip.open(p, "rb") as f:
        f.readline(); seq = f.read().replace(b"\n", b"").upper()
    seq = seq.translate(None, b"N")
    assert set(seq[:1_000_000]) <= set(b"ACGT"), set(seq[:1000])
    write("dna", seq[:NT], seq[NT:NT + NV])
if "image" in NAMES:
    p = fetch("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "image", "cifar-10-binary.tar.gz")
    def imgs(files):
        out = bytearray()
        with tarfile.open(p, "r:gz") as t:
            for fn in files:
                raw = np.frombuffer(t.extractfile(f"cifar-10-batches-bin/{fn}").read(), dtype=np.uint8).reshape(-1, 3073)
                out += raw[:, 1:].reshape(-1, 3, 1024).transpose(0, 2, 1).tobytes()  # drop label; planar RGB -> per-pixel RGB
        return out
    write("image", imgs([f"data_batch_{i}.bin" for i in range(1, 6)]), imgs(["test_batch.bin"]))
if "audio" in NAMES:
    p = fetch("http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", "audio", "mini_speech_commands.zip")
    def mulaw(x):  # x int16 -> uint8 mu-law (mu=255), as in WaveNet
        f = x.astype(np.float32) / 32768.0; y = np.sign(f) * np.log1p(255 * np.abs(f)) / np.log1p(255)
        return ((y + 1) / 2 * 255 + 0.5).astype(np.uint8)
    with zipfile.ZipFile(p) as z:
        wavs = sorted(n for n in z.namelist() if n.endswith(".wav") and "__MACOSX" not in n)
        rng = np.random.default_rng(0); order = rng.permutation(len(wavs))
        val_ids, tr_ids = order[:200], order[200:]
        def clips(ids, need):
            out = bytearray()
            for i in ids:
                with wave.open(io.BytesIO(z.read(wavs[i]))) as w:
                    assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000
                    out += mulaw(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")).tobytes()
                if len(out) >= need: break
            return out
        write("audio", clips(tr_ids, NT), clips(val_ids, NV))
print("ALL DONE")
