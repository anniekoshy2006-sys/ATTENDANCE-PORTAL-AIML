"""
train.py - build the face-recognition model from the dataset.

    python train.py                 # best available back-end (auto)
    python train.py --backend lbph  # force the fully-offline back-end
"""
import argparse
import os
import sys

from face_engine import FaceEngine, HERE

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "sface", "lbph"], default="auto")
    ap.add_argument("--dataset", default=os.path.join(HERE, "dataset"))
    a = ap.parse_args()

    train_dir = os.path.join(a.dataset, "train")
    val_dir = os.path.join(a.dataset, "validation")
    print(f"Training on: {train_dir}")
    eng = FaceEngine(prefer=a.backend)
    rep = eng.train(train_dir, val_dir)
    print(f"\nBack-end used : {rep['backend']}")
    print(f"Images used   : {rep['images_used']}  (skipped: {len(rep['images_skipped'])})")
    for s in rep["images_skipped"]:
        print("   no face found in", s)
    v = rep.get("validation")
    if v:
        print(f"Validation    : {v['correct']}/{v['total']} correct  ({v['accuracy']} %)")
    print("Model saved in ./models  -  now run:  python app.py")
