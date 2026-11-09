import sys, json, time, torch; sys.path.insert(0, ".")
from hsnlm import HSNLM, GPT, Config, CharData, train, evaluate_bpc
torch.set_num_threads(8)
steps = int(sys.argv[1]); which = sys.argv[2].split(",")
data = CharData("data/shakespeare.txt", ctx=128)
CFG = {"gpt": (GPT, Config(vocab=data.V, mlp_mult=4.0)), "hsn": (HSNLM, Config(vocab=data.V, mlp_mult=3.5))}
res = {}
for name in which:
    arch, mode = name.split("-"); cls, cfg = CFG[arch]
    torch.manual_seed(0); m = cls(cfg)
    print(f"== {name}  params={m.n_params()/1e6:.3f}M", flush=True)
    hist, t = train(m, data, steps, mode="backprop" if mode == "bp" else "local", log=max(steps // 5, 1))
    # inference speed: generate 256 tokens, batch 1 and 16
    sp = {}
    for B in (1, 16):
        p = torch.zeros(B, 1, dtype=torch.long); t0 = time.time(); m.generate(p, 127); sp[B] = B * 127 / (time.time() - t0)
    res[name] = dict(params=m.n_params(), val_bpc=hist[-1][2], train_s=t, train_tok_s=steps * 32 * 128 / t, gen_tok_s_b1=sp[1], gen_tok_s_b16=sp[16])
    print(res[name], flush=True)
    json.dump(res, open(f"benchmarks/results_{sys.argv[3] if len(sys.argv)>3 else 'run'}.json", "w"), indent=1)
