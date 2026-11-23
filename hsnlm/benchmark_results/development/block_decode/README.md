# Block / self-speculative decoding primitive

**Status:** exploratory  ·  **Paper:** not in paper (development history)  ·  **Script(s):** [`gpu_blockdecode.py`](../../../benchmarks/gpu_blockdecode.py)

## What it tests and how (from the script)

```text
Tests the throughput PRIMITIVE behind block/self-speculative decoding (RESEARCH_V2.md section 5),
without implementing a full draft-and-verify pipeline or an actual trained diffusion draft model: does
processing a block of b positions in ONE forward pass (inference only, no backward -- this is what a
'verify this drafted block' or 'decode a block of masked positions' pass would cost) get cheaper per
token as b grows, and does that amortization differ between GPT and the sheaf-delta architectures?
This is honest about what it does NOT test: no draft model, no acceptance rate, no actual speculative
decoding loop -- just the raw cost-per-token of processing b positions at once vs one at a time,
which is the mechanism block/speculative decoding would need to pay off.
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python benchmarks/gpu_blockdecode.py
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_blockdecode.json`](gpu_blockdecode.json) | results (machine-readable) | 1.4 KB |

## Results: `gpu_blockdecode.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | tok_s |
|---|---|
| `gpt|1` | 81 |
| `gpt|8` | 608 |
| `gpt|32` | 1,806 |
| `gpt|128` | 4,368 |
| `gpt|512` | 5,513 |
| `gpt|2048` | 5,351 |
| `hsn|1` | 22 |
| `hsn|8` | 175 |
| `hsn|32` | 692 |
| `hsn|128` | 2,542 |
| `hsn|512` | 4,728 |
| `hsn|2048` | 4,989 |
| `hybrid|1` | 28 |
| `hybrid|8` | 218 |
| `hybrid|32` | 868 |
| `hybrid|128` | 2,794 |
| `hybrid|512` | 4,331 |
| `hybrid|2048` | 5,084 |
