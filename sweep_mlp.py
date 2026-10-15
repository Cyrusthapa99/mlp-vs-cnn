"""Hyper-parameter sweep for the MLP (one change at a time), scored on the validation set."""
import json, os, time
from common import *; from models import *
df, X, y, sp = get_data(); tr, va = sp["train"], sp["val"]
BASE = dict(arch="mlp", hidden=512, dropout=0.3, lr=1e-3, wd=1e-4, bs=64, aug=False, cw=0.5, epochs=30)
VAR = [("baseline", {}), ("hidden=256", dict(hidden=256)), ("hidden=1024", dict(hidden=1024)), ("dropout=0.0", dict(dropout=0.0)), ("dropout=0.5", dict(dropout=0.5)),
       ("lr=3e-4", dict(lr=3e-4)), ("lr=3e-3", dict(lr=3e-3)), ("augmentation on", dict(aug=True)), ("class weights off", dict(cw=0.0)),
       ("class weights full", dict(cw=1.0)), ("epochs=60", dict(epochs=60)), ("weight decay=1e-2", dict(wd=1e-2))]
out = json.load(open("mlp_sweep.json")) if os.path.exists("mlp_sweep.json") else {}
logf = open("mlp_sweep.log", "a")
def log(m): logf.write(m + "\n"); logf.flush()
for name, ch in VAR:
    if name in out: continue
    cfg = {**BASE, **ch}; t = time.time(); log(f"== {name} {cfg}")
    m, mean, std, hist, f1, ep = train(cfg, X[tr], y[tr], X[va], y[va], seed=42)
    r = evaluate(y[va].numpy(), predict_proba(m, X[va], mean, std)); r.pop("auc_per_class")
    out[name] = dict(cfg=cfg, best_epoch=ep, secs=time.time() - t, **{f"val_{k}": v for k, v in r.items()})
    json.dump(out, open("mlp_sweep.json", "w")); log(f"-> {name}: val_acc={r['acc']:.3f} val_macroF1={r['f1']:.3f} val_AUC={r['auc_macro']:.3f} best_ep={ep} ({time.time()-t:.0f}s)")
log("SWEEP DONE")
