"""Train the NumPy network on a demo or local MNIST IDX files."""
import argparse
import csv
from pathlib import Path
import struct
import numpy as np
from network import Network


def read_idx(images: Path, labels: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_i, raw_l = images.read_bytes(), labels.read_bytes()
    if len(raw_i) < 16 or len(raw_l) < 8:
        raise ValueError("IDX files are truncated")
    magic_i, count, rows, cols = struct.unpack_from(">IIII", raw_i)
    magic_l, label_count = struct.unpack_from(">II", raw_l)
    if magic_i != 2051 or magic_l != 2049 or count != label_count or len(raw_i) != 16 + count*rows*cols or len(raw_l) != 8 + count:
        raise ValueError("Invalid or mismatched IDX files")
    return np.frombuffer(raw_i, dtype=np.uint8, offset=16).reshape(count, rows*cols).astype(np.float64)/255, np.frombuffer(raw_l, dtype=np.uint8, offset=8).astype(np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.demo:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(400,2))
        y = (x[:,0]*x[:,1] > 0).astype(np.int64)
        sizes = [2, 16, 2]
    elif args.images and args.labels:
        x, y = read_idx(args.images, args.labels)
        sizes = [x.shape[1], 64, 10]
    else:
        parser.error("provide --demo or both IDX files")
    model = Network(sizes, seed=5)
    history = model.train(x, y, epochs=args.epochs, batch_size=64)
    for row in history:
        print(f"epoch {row['epoch']}: loss={row['loss']:.4f} accuracy={row['accuracy']:.3f}")
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch","loss","accuracy"])
            writer.writeheader()
            writer.writerows(history)

