# Six extra domains (code, Hindi, Chinese, DNA, images, audio), 3 seeds

**Status:** current  ·  **Paper:** Table 6 (tab:domains), Fig. 3  ·  **Script(s):** [`gpu_domains_3seed.py`](../../../benchmarks/gpu_domains_3seed.py), [`data_prep/build_domains.py`](../../../benchmarks/data_prep/build_domains.py)

## Summary

All four model families plus the RoPE-DEEPA diagnostic share one JSON (keys domain|model|seed).

## What it tests and how (from the script)

```text
B3 -- multi-modal / multi-domain byte-level LM: the paper's 3-seed, 1,500-step compiled comparison repeated on six
non-enwik8 domains (code, Hindi, Chinese, DNA, images, audio; data/domain_<name>.bytes, built by
benchmarks/data_prep/build_domains.py). Identical models/recipe to gpu_wikitext103_3seed.py, except ALiBi attention uses the
exact fused-kernel path (deepa_common.ALIBI_IMPL="flash"; validated to 4e-5 bpb of the dense-mask path) for BOTH ALiBi models.
For image bytes, bits/byte == bits/dim (the standard image-likelihood metric); for 8-bit mu-law audio it is bits/sample.
Eval: val_bpb = 64 windows @1024 (repo convention), val_bpb_full = all 976 windows @1024, plus @2048/@4096 (32 windows).
Usage: python benchmarks/gpu_domains_3seed.py [domain ...] [--kinds=a,b]   (default: all six domains, the 3 paper models)
```

## Reproduce

From the `hsnlm/` root (GPU runs need CUDA; see the top-level README for the environment):

```bash
python -I benchmarks/data_prep/build_domains.py .
python benchmarks/gpu_domains_3seed.py                                   # DEEPA, Pure ALiBi, RoPE GPT
python benchmarks/gpu_domains_3seed.py --kinds=sheafdelta-pure            # Pure SheafDelta ablation
python benchmarks/gpu_domains_3seed.py image code --kinds=rope-hybrid-h6  # RoPE-DEEPA diagnostic
```

## Files

| file | kind | size |
|---|---|---|
| [`gpu_domains_3seed.json`](gpu_domains_3seed.json) | results (machine-readable) | 52.9 KB |
| [`gpu_domains_3seed.log`](gpu_domains_3seed.log) | stdout of the run | 22.4 KB |
| [`gpu_domains_ropehybrid.log`](gpu_domains_ropehybrid.log) | stdout of the run | 2.9 KB |
| [`gpu_domains_sheafpure.log`](gpu_domains_sheafpure.log) | stdout of the run | 8.1 KB |

## Results: `gpu_domains_3seed.json`

Auto-extracted per-run metrics (bits/byte lower is better; accuracies in [0,1]; `x` = not solved). Paper tables aggregate these over seeds.

| run | val_bpb_full | val_bpb | val_bpb_len2048 | val_bpb_len4096 | tok_s | peak_mem_mb | train_s |
|---|---|---|---|---|---|---|---|
| `code|alibi-hybrid-h6|seed0` | 1.1497 | 1.3030 | 1.2441 | 1.1227 | 86,684.9 | 2,746.6 | 283.5 |
| `code|alibi-pure|seed0` | 1.1860 | 1.3423 | 1.2683 | 1.1399 | 117,009.2 | 2,285.8 | 210.0 |
| `code|rope-gpt|seed0` | 1.2237 | 1.3819 | 2.1296 | 3.0492 | 98,570.4 | 2,409.1 | 249.3 |
| `code|alibi-hybrid-h6|seed1` | 1.1375 | 1.2766 | 1.2173 | 1.1252 | 86,612.5 | 2,749.7 | 283.7 |
| `code|alibi-pure|seed1` | 1.1940 | 1.3499 | 1.2767 | 1.1618 | 116,967.8 | 2,284.8 | 210.1 |
| `code|rope-gpt|seed1` | 1.2077 | 1.3677 | 1.9964 | 3.1298 | 98,508.8 | 2,409.1 | 249.5 |
| `code|alibi-hybrid-h6|seed2` | 1.1377 | 1.2806 | 1.2210 | 1.1013 | 86,538.1 | 2,749.7 | 284.0 |
| `code|alibi-pure|seed2` | 1.1848 | 1.3456 | 1.2707 | 1.1405 | 116,996.7 | 2,284.8 | 210.1 |
| `code|rope-gpt|seed2` | 1.2079 | 1.3630 | 1.9659 | 2.8211 | 98,480.2 | 2,409.1 | 249.6 |
| `hindi|alibi-hybrid-h6|seed0` | 0.8756 | 0.9419 | 0.9367 | 0.9006 | 86,548.2 | 2,749.7 | 284.0 |
| `hindi|alibi-pure|seed0` | 0.9490 | 1.0266 | 1.0217 | 0.9838 | 116,723.8 | 2,284.8 | 210.5 |
| `hindi|rope-gpt|seed0` | 0.9719 | 1.0540 | 1.2685 | 1.6656 | 98,285.6 | 2,409.1 | 250.0 |
| `hindi|alibi-hybrid-h6|seed1` | 0.8757 | 0.9327 | 0.9273 | 0.8953 | 86,569.8 | 2,749.7 | 283.9 |
| `hindi|alibi-pure|seed1` | 0.9604 | 1.0302 | 1.0253 | 0.9945 | 116,597.7 | 2,284.8 | 210.8 |
| `hindi|rope-gpt|seed1` | 1.0325 | 1.1254 | 1.3786 | 1.8010 | 98,148.9 | 2,409.1 | 250.4 |
| `hindi|alibi-hybrid-h6|seed2` | 0.8723 | 0.9330 | 0.9276 | 0.8966 | 86,486.2 | 2,749.7 | 284.2 |
| `hindi|alibi-pure|seed2` | 0.9665 | 1.0427 | 1.0380 | 1.0037 | 116,668.2 | 2,284.8 | 210.6 |
| `hindi|rope-gpt|seed2` | 1.0200 | 1.1065 | 1.2609 | 1.5806 | 98,192.0 | 2,409.1 | 250.3 |
| `zh|alibi-hybrid-h6|seed0` | 2.0536 | 2.3454 | 2.3335 | 2.3557 | 86,562.1 | 2,749.7 | 283.9 |
| `zh|alibi-pure|seed0` | 2.0348 | 2.3106 | 2.2668 | 2.2847 | 116,857.3 | 2,284.8 | 210.3 |
| `zh|rope-gpt|seed0` | 2.2707 | 2.5058 | 3.0102 | 3.8164 | 98,324.2 | 2,409.1 | 249.9 |
| `zh|alibi-hybrid-h6|seed1` | 2.0529 | 2.3445 | 2.3324 | 2.3539 | 86,501.1 | 2,749.7 | 284.1 |
| `zh|alibi-pure|seed1` | 2.0174 | 2.2907 | 2.2433 | 2.2575 | 116,793.0 | 2,284.8 | 210.4 |
| `zh|rope-gpt|seed1` | 2.2610 | 2.5035 | 3.1914 | 4.2340 | 98,370.5 | 2,409.1 | 249.8 |
| `zh|alibi-hybrid-h6|seed2` | 2.0678 | 2.3451 | 2.3341 | 2.3622 | 86,565.9 | 2,749.7 | 283.9 |
| `zh|alibi-pure|seed2` | 2.0228 | 2.2917 | 2.2444 | 2.2566 | 116,910.1 | 2,284.8 | 210.2 |
| `zh|rope-gpt|seed2` | 2.2533 | 2.4924 | 3.0008 | 3.7459 | 98,370.1 | 2,409.1 | 249.8 |
| `dna|alibi-hybrid-h6|seed0` | 1.7417 | 1.7619 | 1.7605 | 1.7734 | 86,492.7 | 2,749.7 | 284.1 |
| `dna|alibi-pure|seed0` | 1.8265 | 1.8270 | 1.8247 | 1.8409 | 116,977.0 | 2,284.8 | 210.1 |
| `dna|rope-gpt|seed0` | 1.7854 | 1.7856 | 1.8497 | 1.8907 | 98,346.2 | 2,409.1 | 249.9 |
| `dna|alibi-hybrid-h6|seed1` | 1.7420 | 1.7632 | 1.7620 | 1.7753 | 86,391.4 | 2,749.7 | 284.5 |
| `dna|alibi-pure|seed1` | 1.8382 | 1.8372 | 1.8348 | 1.8497 | 117,005.6 | 2,284.8 | 210.0 |
| `dna|rope-gpt|seed1` | 1.7902 | 1.7907 | 1.8357 | 1.8719 | 98,430.2 | 2,409.1 | 249.7 |
| `dna|alibi-hybrid-h6|seed2` | 1.7423 | 1.7612 | 1.7597 | 1.7734 | 86,418.0 | 2,749.7 | 284.4 |
| `dna|alibi-pure|seed2` | 1.8522 | 1.8506 | 1.8486 | 1.8615 | 116,996.0 | 2,284.8 | 210.1 |
| `dna|rope-gpt|seed2` | 1.7889 | 1.7914 | 1.8492 | 1.8894 | 98,419.9 | 2,409.1 | 249.7 |
| `image|alibi-hybrid-h6|seed0` | 4.4189 | 4.6956 | 4.6909 | 4.7034 | 86,542.6 | 2,749.7 | 284.0 |
| `image|alibi-pure|seed0` | 4.6518 | 4.9047 | 4.8900 | 4.8952 | 117,030.6 | 2,284.8 | 210.0 |
| `image|rope-gpt|seed0` | 3.9768 | 4.1741 | 5.0634 | 5.9508 | 98,655.0 | 2,409.1 | 249.1 |
| `image|alibi-hybrid-h6|seed1` | 4.4466 | 4.7227 | 4.7162 | 4.7291 | 86,525.4 | 2,749.7 | 284.0 |
| `image|alibi-pure|seed1` | 4.6485 | 4.9095 | 4.8915 | 4.8894 | 117,015.3 | 2,284.8 | 210.0 |
| `image|rope-gpt|seed1` | 3.9654 | 4.1681 | 4.9177 | 6.1454 | 98,786.7 | 2,409.1 | 248.8 |
| `image|alibi-hybrid-h6|seed2` | 4.4490 | 4.7195 | 4.7131 | 4.7221 | 85,827.2 | 2,749.7 | 286.3 |
| `image|alibi-pure|seed2` | 4.6707 | 4.9281 | 4.9189 | 4.9242 | 117,020.5 | 2,284.8 | 210.0 |
| `image|rope-gpt|seed2` | 3.9562 | 4.1640 | 5.0505 | 6.1384 | 98,695.4 | 2,409.1 | 249.0 |
| `audio|alibi-hybrid-h6|seed0` | 3.3291 | 3.2045 | 3.1993 | 2.9897 | 86,633.5 | 2,749.7 | 283.7 |
| `audio|alibi-pure|seed0` | 3.4595 | 3.3146 | 3.3073 | 3.0926 | 117,219.4 | 2,284.8 | 209.7 |
| `audio|rope-gpt|seed0` | 3.4289 | 3.2870 | 3.5900 | 3.9549 | 98,653.2 | 2,409.1 | 249.1 |
| `audio|alibi-hybrid-h6|seed1` | 3.3253 | 3.1977 | 3.1921 | 2.9820 | 86,671.6 | 2,749.7 | 283.6 |
| `audio|alibi-pure|seed1` | 3.4362 | 3.2988 | 3.2902 | 3.0754 | 117,176.1 | 2,284.8 | 209.7 |
| `audio|rope-gpt|seed1` | 3.3906 | 3.2542 | 3.5997 | 3.9668 | 98,595.2 | 2,409.1 | 249.3 |
| `audio|alibi-hybrid-h6|seed2` | 3.3357 | 3.2081 | 3.2031 | 2.9939 | 86,691.6 | 2,749.7 | 283.5 |
| `audio|alibi-pure|seed2` | 3.4352 | 3.2926 | 3.2841 | 3.0732 | 117,244.9 | 2,284.8 | 209.6 |
| `audio|rope-gpt|seed2` | 3.4153 | 3.2708 | 3.4236 | 3.8620 | 98,614.1 | 2,409.1 | 249.2 |
| `image|rope-hybrid-h6|seed0` | 3.9418 | 4.1782 | 4.4433 | 5.0601 | 83,464.6 | 2,763.3 | 294.4 |
| `image|rope-hybrid-h6|seed1` | 3.9383 | 4.1756 | 4.6140 | 5.2296 | 83,513.0 | 2,759.7 | 294.3 |
| `image|rope-hybrid-h6|seed2` | 3.9607 | 4.1929 | 4.6134 | 5.3564 | 83,467.7 | 2,759.7 | 294.4 |
| `code|rope-hybrid-h6|seed0` | 1.1395 | 1.2879 | 1.2303 | 1.1295 | 83,514.9 | 2,761.7 | 294.3 |
| `code|rope-hybrid-h6|seed1` | 1.1477 | 1.2997 | 1.2455 | 1.1413 | 83,578.3 | 2,761.7 | 294.0 |
| `code|rope-hybrid-h6|seed2` | 1.1480 | 1.2949 | 1.2449 | 1.1384 | 83,583.1 | 2,761.7 | 294.0 |
| `code|sheafdelta-pure|seed0` | 1.3881 | 1.5757 | 1.5528 | 1.4207 | 80,104.9 | 2,946.4 | 306.8 |
| `code|sheafdelta-pure|seed1` | 1.3673 | 1.5496 | 1.5250 | 1.3899 | 80,116.8 | 2,949.1 | 306.8 |
| `code|sheafdelta-pure|seed2` | 1.3682 | 1.5390 | 1.5136 | 1.3793 | 80,109.7 | 2,949.1 | 306.8 |
| `hindi|sheafdelta-pure|seed0` | 0.8774 | 0.9404 | 0.9355 | 0.9036 | 80,093.4 | 2,949.1 | 306.8 |
| `hindi|sheafdelta-pure|seed1` | 0.8761 | 0.9349 | 0.9293 | 0.8993 | 80,068.0 | 2,949.1 | 306.9 |
| `hindi|sheafdelta-pure|seed2` | 0.8764 | 0.9372 | 0.9320 | 0.9009 | 80,064.4 | 2,949.1 | 307.0 |
| `zh|sheafdelta-pure|seed0` | 2.0851 | 2.3535 | 2.3413 | 2.3667 | 80,133.7 | 2,949.1 | 306.7 |
| `zh|sheafdelta-pure|seed1` | 2.0796 | 2.3583 | 2.3461 | 2.3744 | 80,147.6 | 2,949.1 | 306.6 |
| `zh|sheafdelta-pure|seed2` | 2.0851 | 2.3572 | 2.3453 | 2.3727 | 80,129.7 | 2,949.1 | 306.7 |
| `dna|sheafdelta-pure|seed0` | 1.7389 | 1.7601 | 1.7589 | 1.7720 | 80,010.8 | 2,949.1 | 307.2 |
| `dna|sheafdelta-pure|seed1` | 1.7394 | 1.7595 | 1.7583 | 1.7719 | 80,024.9 | 2,949.1 | 307.1 |
| `dna|sheafdelta-pure|seed2` | 1.7370 | 1.7589 | 1.7581 | 1.7708 | 80,021.7 | 2,949.1 | 307.1 |
| `image|sheafdelta-pure|seed0` | 4.4770 | 4.7597 | 4.7559 | 4.7658 | 80,039.0 | 2,949.1 | 307.1 |
| `image|sheafdelta-pure|seed1` | 4.4995 | 4.7823 | 4.7804 | 4.7897 | 80,057.9 | 2,949.1 | 307.0 |
| `image|sheafdelta-pure|seed2` | 4.5046 | 4.7828 | 4.7796 | 4.7922 | 80,053.6 | 2,949.1 | 307.0 |
| `audio|sheafdelta-pure|seed0` | 3.3368 | 3.2117 | 3.2062 | 2.9938 | 80,098.6 | 2,949.1 | 306.8 |
| `audio|sheafdelta-pure|seed1` | 3.3234 | 3.1935 | 3.1874 | 2.9790 | 80,071.4 | 2,949.1 | 306.9 |
| `audio|sheafdelta-pure|seed2` | 3.3330 | 3.2082 | 3.2023 | 2.9899 | 80,105.9 | 2,949.1 | 306.8 |
