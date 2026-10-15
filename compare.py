"""Train the MLP and the CNN (3 seeds each) and compare them on the held-out test blocks:
confusion matrices, precision, recall, F1, and AUC-ROC (one-vs-rest)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_fscore_support, auc as auc_fn
from common import *; from models import *
df, X, y, sp = get_data(); tr, va, te = sp["train"], sp["val"], sp["test"]; yt = y[te].numpy(); labs = np.unique(yt)
names = df.drop_duplicates("cls").set_index("cls").name.to_dict(); os.makedirs("results", exist_ok=True)
sw = json.load(open("mlp_sweep.json")); win = max(sw, key=lambda n: sw[n]["val_f1"])
CFG = {"MLP": sw[win]["cfg"],
       "CNN": dict(arch="cnn", width=16, dropout=0.3, lr=3e-3, wd=1e-4, bs=64, aug=False, cw=0.5, epochs=25)}   # CNN settings selected in Assignment 7
logf = open("final.log", "a")
def log(m): print(m); logf.write(m + "\n"); logf.flush()
log(f"MLP configuration chosen on validation: {win} {CFG['MLP']} (val macro F1 {sw[win]['val_f1']:.3f})")
res = {k: [] for k in CFG}; keep = {}
for name, cfg in CFG.items():
    for seed in (42, 43, 44):
        m, mean, std, hist, f1, ep = train(cfg, X[tr], y[tr], X[va], y[va], seed=seed); P = predict_proba(m, X[te], mean, std); r = evaluate(yt, P)
        res[name].append(dict(seed=seed, best_epoch=ep, **r)); keep[(name, seed)] = (P, hist)
        log(f"{name} seed {seed}: acc={r['acc']:.4f} macroP={r['precision']:.4f} macroR={r['recall']:.4f} macroF1={r['f1']:.4f} AUC={r['auc_macro']:.4f} (best epoch {ep})")
keys = ["acc", "precision", "recall", "f1", "auc_macro"]
S = {n: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in keys} for n, rs in res.items()}
for n in S: log(f"{n} MEAN+-SD: " + ", ".join(f"{k}={v[0]:.4f}+-{v[1]:.4f}" for k, v in S[n].items()))

COL = {"MLP": "tab:orange", "CNN": "tab:blue"}
# ---- confusion matrices + per-class tables (seed 42) ----
for n in CFG:
    P = keep[(n, 42)][0]; pt = P.argmax(1); cols = sorted(set(labs) | set(pt)); cm = confusion_matrix(yt, pt, labels=cols)[[cols.index(c) for c in labs]]
    fig, ax = plt.subplots(figsize=(10, 6.5)); ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7, rotation=90); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("Predicted ClassId"); ax.set_ylabel("True ClassId"); ax.set_title(f"{n}: test confusion matrix (seed 42, {len(yt)} images)"); plt.tight_layout(); plt.savefig(f"results/confusion_matrix_{n.lower()}.png", dpi=130); plt.close()
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=labs, zero_division=0); auc = res[n][0]["auc_per_class"]
    pd.DataFrame(dict(ClassId=labs, Name=[names[c] for c in labs], precision=p, recall=r, f1=f, auc=[auc[int(c)] for c in labs], support=s)).to_csv(f"results/per_class_{n.lower()}.csv", index=False)

# ---- ROC curves (one-vs-rest, seed 42): macro- and micro-average ----
grid = np.linspace(0, 1, 201); fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for n in CFG:
    P = keep[(n, 42)][0]; tprs = []
    for c in labs:
        fpr, tpr, _ = roc_curve((yt == c).astype(int), P[:, c]); tprs.append(np.interp(grid, fpr, tpr))
    ax[0].plot(grid, np.mean(tprs, 0), color=COL[n], lw=2, label=f"{n} (macro AUC = {res[n][0]['auc_macro']:.3f})")
    Y = (yt[:, None] == labs[None, :]).astype(int); fpr, tpr, _ = roc_curve(Y.ravel(), P[:, labs].ravel())
    ax[1].plot(fpr, tpr, color=COL[n], lw=2, label=f"{n} (micro AUC = {auc_fn(fpr, tpr):.3f})")
for a, t in zip(ax, ["Macro-average ROC (one-vs-rest)", "Micro-average ROC (one-vs-rest)"]):
    a.plot([0, 1], [0, 1], "k--", lw=1, label="Chance"); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.set_title(t); a.legend(loc="lower right"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/roc_curves.png", dpi=130); plt.close()

# ---- per-class AUC, metric bars, learning curves ----
fig, ax = plt.subplots(figsize=(11, 3.8)); xx = np.arange(len(labs)); w = .38
for k, n in enumerate(CFG): ax.bar(xx + (k - .5) * w, [res[n][0]["auc_per_class"][int(c)] for c in labs], w, label=n, color=COL[n])
ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.05); ax.set_xlabel("ClassId"); ax.set_ylabel("AUC-ROC"); ax.legend(); ax.set_title("Per-class AUC-ROC on the test blocks (seed 42)")
plt.tight_layout(); plt.savefig("results/per_class_auc.png", dpi=130); plt.close()
fig, ax = plt.subplots(figsize=(8, 4)); lab = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]; xx = np.arange(5)
for k, n in enumerate(CFG):
    ax.bar(xx + (k - .5) * w, [S[n][q][0] for q in keys], w, yerr=[S[n][q][1] for q in keys], capsize=3, label=n, color=COL[n])
    for i, q in enumerate(keys): ax.text(i + (k - .5) * w, S[n][q][0] + S[n][q][1] + .01, f"{S[n][q][0]:.2f}", ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(lab); ax.set_ylim(0, 1.1); ax.set_title("Test metrics, mean ± SD over 3 seeds (macro averages)"); ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig("results/metrics_comparison.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for n in CFG:
    h = pd.DataFrame(keep[(n, 42)][1]); ax[0].plot(h.epoch, h.loss, color=COL[n], label=n); ax[1].plot(h.epoch, h.val_f1, color=COL[n], label=n)
ax[0].set_title("Training loss (seed 42)"); ax[1].set_title("Validation macro F1 (seed 42)")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/learning_curves.png", dpi=130); plt.close()
json.dump(dict(cfg=CFG, mean_sd=S, runs={n: [{k: v for k, v in r.items() if k != "auc_per_class"} for r in rs] for n, rs in res.items()}, n_test=int(len(yt)), n_classes=int(len(labs))), open("results/summary.json", "w"), indent=2)
log("FINAL DONE")
