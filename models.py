"""Two classifiers for the same data: a fully connected network (MLP) and a convolutional network (CNN)."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

class MLP(nn.Module):
    """Neural network: flattened 64x64x3 pixels -> hidden -> hidden/2 -> 58 classes (no convolution)."""
    def __init__(self, n_classes=58, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64 * 64 * 3, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden // 2, n_classes))
    def forward(self, x): return self.net(x)

class CNN(nn.Module):
    """Deep learning: 3 x (conv-BN-ReLU-pool) 64->32->16->8, then 2 fully connected layers (the Assignment 7 network)."""
    def __init__(self, n_classes=58, width=16, dropout=0.3, hidden=256):
        super().__init__()
        def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, width), block(width, 2 * width), block(2 * width, 4 * width))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * width * 8 * 8, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    def forward(self, x): return self.head(self.features(x))

def build(cfg):
    return MLP(58, cfg["hidden"], cfg["dropout"]) if cfg["arch"] == "mlp" else CNN(58, cfg["width"], cfg["dropout"])

def augment(x):
    B = x.size(0); a = (torch.rand(B) * 2 - 1) * math.radians(15); s = 1 / (0.9 + 0.2 * torch.rand(B))
    th = torch.zeros(B, 2, 3); th[:, 0, 0] = s * torch.cos(a); th[:, 0, 1] = -s * torch.sin(a); th[:, 1, 0] = s * torch.sin(a); th[:, 1, 1] = s * torch.cos(a)
    th[:, :, 2] = (torch.rand(B, 2) * 2 - 1) * 0.16
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    br = 1 + (torch.rand(B, 1, 1, 1) - 0.5) * 0.5; ct = 1 + (torch.rand(B, 1, 1, 1) - 0.5) * 0.5
    m = x.mean((1, 2, 3), keepdim=True); return ((x - m) * ct + m * br).clamp(0, 1)

def class_weights(y, n_classes, power):
    cnt = np.bincount(y, minlength=n_classes).astype(float); w = np.where(cnt > 0, (cnt.sum() / (n_classes * np.maximum(cnt, 1))) ** power, 0.0)
    w = w / w[cnt > 0].mean(); w[cnt == 0] = 1.0; return torch.tensor(w, dtype=torch.float32)

def predict_proba(model, X, mean, std, bs=256):
    """Softmax probabilities (N x 58) - needed for the ROC / AUC."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs): out.append(F.softmax(model((X[i:i + bs] - mean) / std), 1))
    return torch.cat(out).numpy()

def evaluate(y, P):
    """Precision, recall, F1 (macro + weighted) and one-vs-rest AUC-ROC over the classes present in `y`."""
    pred = P.argmax(1); labs = np.unique(y); out = dict(acc=float(accuracy_score(y, pred)))
    for avg, tag in (("macro", ""), ("weighted", "w_")):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labs, average=avg, zero_division=0)
        out.update({f"{tag}precision": float(p), f"{tag}recall": float(r), f"{tag}f1": float(f)})
    aucs = {int(c): float(roc_auc_score((y == c).astype(int), P[:, c])) for c in labs}   # one-vs-rest AUC for each class
    out["auc_macro"] = float(np.mean(list(aucs.values()))); out["auc_per_class"] = aucs
    return out

def train(cfg, Xtr, ytr, Xva, yva, seed=42, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mean = Xtr.mean((0, 2, 3), keepdim=True); std = Xtr.std((0, 2, 3), keepdim=True)
    model = build(cfg); opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * math.ceil(len(Xtr) / cfg["bs"]); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps)
    crit = nn.CrossEntropyLoss(weight=class_weights(ytr.numpy(), 58, cfg["cw"]) if cfg["cw"] > 0 else None)
    hist = []; best = (-1, None, 0)
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0
        for i in range(0, len(Xtr), cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            if len(idx) < 2: continue
            xb = Xtr[idx]; xb = augment(xb) if cfg["aug"] else xb
            opt.zero_grad(); loss = crit(model((xb - mean) / std), ytr[idx]); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        m = evaluate(yva.numpy(), predict_proba(model, Xva, mean, std))
        hist.append(dict(epoch=ep, loss=tot / len(Xtr), val_acc=m["acc"], val_f1=m["f1"], val_auc=m["auc_macro"]))
        if log: log(f"  ep{ep:02d} loss={tot/len(Xtr):.4f} val_acc={m['acc']:.3f} val_macroF1={m['f1']:.3f}")
        if m["f1"] > best[0]: best = (m["f1"], {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); return model, mean, std, hist, best[0], best[2]
