"""Data preparation for the Indian Traffic Sign dataset (Week 2 plan):
MD5 de-duplication -> scene (physical sign) inference -> leakage-aware group split."""
import os, re, hashlib, random
import numpy as np, pandas as pd
from PIL import Image

NAME_RE = re.compile(r"^(\d{3})(?:_(\d+))?_(\d{4})(?:_\d+)?(?:_j)?\.png$")
BLOCK = 20   # frames per block: neighbouring frames of one physical sign stay together
MARGIN = 2   # train frames this close to a val/test block edge are purged (boundary leakage)

def build_metadata(roots, labels_csv):
    """One row per image file. `roots` = folders holding <ClassId>/<image>.png"""
    names = pd.read_csv(labels_csv).set_index("ClassId")["Name"].to_dict()
    rows = []
    for root in roots:
        for cls in sorted(os.listdir(root), key=int):
            for f in sorted(os.listdir(os.path.join(root, cls))):
                m = NAME_RE.match(f)
                frame = int(m.group(3)) if m else -1
                rows.append(dict(path=os.path.join(root, cls, f), file=f, cls=int(cls),
                                 name=names[int(cls)], frame=frame, scene=f"{int(cls)}:{frame // BLOCK}"))
    df = pd.DataFrame(rows)
    df["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df.path]
    return df

def dedupe(df):
    return df.drop_duplicates("md5").reset_index(drop=True)

def group_split(df, seed=42, min_scenes=3):
    """Classes with >= min_scenes physical scenes: ~60/20/20 split BY SCENE.
    Classes with fewer scenes stay in training (low-resource classes)."""
    rng = random.Random(seed); split = {}
    for c, g in df.groupby("cls"):
        scenes = sorted(g.scene.unique())
        if len(scenes) < min_scenes:
            for s in scenes: split[s] = "train"
            continue
        rng.shuffle(scenes)
        n = len(scenes); n_te = max(1, round(0.2 * n)); n_va = max(1, round(0.2 * n))
        for i, s in enumerate(scenes):
            split[s] = "test" if i < n_te else "val" if i < n_te + n_va else "train"
    df = df.copy(); df["split"] = df.scene.map(split)
    # purge: drop train images within MARGIN frames of a val/test image of the same class
    drop = np.zeros(len(df), bool)
    for c, g in df.groupby("cls"):
        held = g[g.split != "train"].frame.values
        if len(held) == 0: continue
        for i, fr in zip(g.index[g.split == "train"], g[g.split == "train"].frame.values):
            if np.abs(held - fr).min() <= MARGIN: drop[df.index.get_loc(i)] = True
    df.attrs["purged"] = int(drop.sum())
    return df[~drop].reset_index(drop=True)

def load_array(paths, size=64):
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)) for p in paths])

if __name__ == "__main__":
    import os
    roots = os.environ.get("DATA_DIRS", "DATA").split(","); labels = os.environ.get("LABELS_CSV", "labels.csv")
    df = build_metadata(roots, labels)
    print("files", len(df), "classes", df.cls.nunique())
    u = dedupe(df); print("unique MD5", len(u), "duplicates", len(df) - len(u))
    s = group_split(u)
    for k in ["train", "val", "test"]:
        g = s[s.split == k]; print(k, len(g), "images", g.cls.nunique(), "classes", g.scene.nunique(), "blocks")
    print("purged", s.attrs.get("purged"))
