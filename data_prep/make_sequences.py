# data_prep/make_sequences.py
import numpy as np
import os, glob
from sklearn.preprocessing import LabelEncoder
import argparse

def load_sequences(folder, seq_len=30):
    X, y = [], []
    classes = sorted(os.listdir(folder))
    for cls in classes:
        cls_folder = os.path.join(folder, cls)
        if not os.path.isdir(cls_folder): continue
        # read per-video frames assuming pattern <vid>_f*.npy
        vids = {}
        for f in os.listdir(cls_folder):
            if f.endswith(".npy"):
                vidname = "_".join(f.split("_")[:-1])  # crude
                vids.setdefault(vidname, []).append(os.path.join(cls_folder, f))
        for vid, frames in vids.items():
            frames.sort()
            seq = []
            for p in frames[:seq_len]:
                seq.append(np.load(p))
            # pad if short
            if len(seq) < seq_len:
                pad = [np.zeros_like(seq[0])]*(seq_len-len(seq))
                seq.extend(pad)
            X.append(np.stack(seq))
            y.append(cls)
    X = np.array(X)  # shape (N, seq_len, feat)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_prefix", default="data")
    parser.add_argument("--seq_len", type=int, default=30)
    args = parser.parse_args()

    X, y, le = load_sequences(args.data_dir, args.seq_len)
    np.save(f"{args.out_prefix}_X.npy", X)
    np.save(f"{args.out_prefix}_y.npy", y)
    import joblib
    joblib.dump(le, f"{args.out_prefix}_labelenc.joblib")
    print("Saved:", X.shape, y.shape)
