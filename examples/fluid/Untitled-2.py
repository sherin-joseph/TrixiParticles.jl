"""
compare_bubble_metrics.py

Postprocess VTU time series referenced by a PVD file to compute:
  - bubble centroid height y_c(t)
  - width W(t), height H(t), aspect ratio AR(t)=W/H
  - rise velocity dy_c/dt (finite difference)

Then compare Newtonian vs Carreau–Yasuda by generating:
  - metrics_newtonian.csv
  - metrics_carreau_yasuda.csv
  - compare_yc.png, compare_ar.png, compare_vel.png

Assumptions:
  1) Your bubble (gas) particles are in ONE dataset series (e.g., "..._fluid_2.pvd"),
     and the corresponding VTU contains ONLY bubble particles (or you filtered it that way).
  2) VTU files use AppendedData encoding="raw" and vtkZLibDataCompressor (common in ParaView VTU).

Usage:
  python compare_bubble_metrics.py \
      --newton /path/to/out_bubble_newton_fluid_2.pvd \
      --cy     /path/to/out_bubble_carreau_yasuda_fluid_2.pvd \
      --outdir /path/to/output_folder

Dependencies:
  pip install numpy pandas matplotlib lxml
"""

import os
import re
import zlib
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt


# -----------------------------
# VTU reading (Points only)
# -----------------------------
def _read_u64(buf: bytes, idx: int) -> Tuple[int, int]:
    return int.from_bytes(buf[idx:idx + 8], byteorder="little", signed=False), idx + 8


def read_vtu_points(filepath: str) -> np.ndarray:
    """
    Read VTU Points DataArray (N x ncomp) from an appended raw, zlib-compressed VTU.
    Returns: points array shaped (N, 3) or (N, 2) depending on NumberOfComponents.
    """
    with open(filepath, "rb") as f:
        data = f.read()

    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(data, parser=parser)

    header_type = root.get("header_type", "UInt32")
    if header_type != "UInt64":
        raise ValueError(f"{os.path.basename(filepath)}: unsupported header_type={header_type} (expected UInt64)")

    compressor = root.get("compressor")
    if compressor != "vtkZLibDataCompressor":
        raise ValueError(f"{os.path.basename(filepath)}: unsupported compressor={compressor}")

    points_da = root.find(".//Points/DataArray")
    if points_da is None:
        raise ValueError(f"{os.path.basename(filepath)}: cannot find Points/DataArray")

    dtype_str = points_da.get("type")
    ncomp = int(points_da.get("NumberOfComponents", "3"))
    offset = int(points_da.get("offset", "0"))

    # locate appended raw data start: <AppendedData encoding="raw"> _  (underscore is the first byte)
    m = re.search(br"<AppendedData[^>]*encoding=\"raw\"[^>]*>\s*_", data)
    if not m:
        raise ValueError(f"{os.path.basename(filepath)}: cannot locate appended raw data start")

    start = m.end()
    endm = re.search(br"</AppendedData>", data[start:])
    if not endm:
        raise ValueError(f"{os.path.basename(filepath)}: cannot locate appended raw data end")

    appended = data[start:start + endm.start()]
    block = appended[offset:]

    # VTU zlib blocks layout for appended data with UInt64 header:
    # [numBlocks][blockSize][lastBlockSize][compressedBlockSizes...][compressedData...]
    idx = 0
    num_blocks, idx = _read_u64(block, idx)
    _, idx = _read_u64(block, idx)  # blockSize (unused)
    _, idx = _read_u64(block, idx)  # lastBlockSize (unused)

    comp_sizes: List[int] = []
    for _ in range(num_blocks):
        cs, idx = _read_u64(block, idx)
        comp_sizes.append(cs)

    comp_data = block[idx:]

    out = bytearray()
    pos = 0
    for cs in comp_sizes:
        chunk = comp_data[pos:pos + cs]
        pos += cs
        out.extend(zlib.decompress(chunk))

    raw = bytes(out)

    if dtype_str == "Float64":
        dtype = np.float64
    elif dtype_str == "Float32":
        dtype = np.float32
    else:
        raise ValueError(f"{os.path.basename(filepath)}: unsupported dtype={dtype_str}")

    arr = np.frombuffer(raw, dtype=dtype)
    pts = arr.reshape((-1, ncomp))
    return pts


# -----------------------------
# Metrics
# -----------------------------
def bubble_metrics_from_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """
    pts: (N, ncomp) where first 2 columns are x,y
    returns: y_c, W, H, AR
    """
    if pts.shape[0] == 0:
        raise ValueError("No points in VTU")

    x = pts[:, 0]
    y = pts[:, 1]
    y_c = float(np.mean(y))
    W = float(np.max(x) - np.min(x))
    H = float(np.max(y) - np.min(y))
    AR = float(W / (H if H != 0 else np.finfo(float).eps))
    return y_c, W, H, AR


def parse_pvd_series(pvd_path: str) -> List[Tuple[float, str]]:
    """
    Returns list of (timestep, vtu_relative_path) from a PVD file.
    """
    with open(pvd_path, "rb") as f:
        root = etree.fromstring(f.read())

    datasets = root.findall(".//DataSet")
    series = []
    for ds in datasets:
        t = float(ds.get("timestep"))
        file_rel = ds.get("file")
        if file_rel is None:
            continue
        series.append((t, file_rel))
    series.sort(key=lambda x: x[0])
    return series


def compute_metrics_from_pvd(pvd_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute time series metrics for all VTUs referenced by PVD.
    Skips missing VTUs, returns (DataFrame, missing_files).
    """
    base_dir = os.path.dirname(os.path.abspath(pvd_path))
    series = parse_pvd_series(pvd_path)

    rows = []
    missing: List[str] = []

    for t, file_rel in series:
        vtu_path = os.path.join(base_dir, file_rel)
        if not os.path.exists(vtu_path):
            missing.append(file_rel)
            continue

        pts = read_vtu_points(vtu_path)
        y_c, W, H, AR = bubble_metrics_from_points(pts)
        rows.append((t, y_c, W, H, AR))

    df = pd.DataFrame(rows, columns=["t", "y_c", "W", "H", "AR"]).sort_values("t").reset_index(drop=True)
    if len(df) >= 2:
        df["dyc_dt"] = df["y_c"].diff() / df["t"].diff()
    else:
        df["dyc_dt"] = np.nan

    return df, missing


# -----------------------------
# Plotting
# -----------------------------
def plot_compare(x1, y1, label1, x2, y2, label2, xlabel, ylabel, outpath):
    plt.figure()
    if len(x1) > 0:
        plt.plot(x1, y1, marker="o", label=label1)
    if len(x2) > 0:
        plt.plot(x2, y2, marker="o", label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--newton", default="out/out_bubble_newton2_fluid_2.pvd", help="Path to Newtonian .pvd (bubble/gas dataset)")
    ap.add_argument("--cy", default="out/out_bubble_carreau_yasuda2_fluid_2.pvd", help="Path to Carreau–Yasuda .pvd (bubble/gas dataset)")
    ap.add_argument("--outdir", default="out/comparison_results", help="Output directory for CSVs/plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Compute metrics
    dfN, missingN = compute_metrics_from_pvd(args.newton)
    dfCY, missingCY = compute_metrics_from_pvd(args.cy)

    # Save CSVs
    csvN = os.path.join(args.outdir, "metrics_newtonian.csv")
    csvCY = os.path.join(args.outdir, "metrics_carreau_yasuda.csv")
    dfN.to_csv(csvN, index=False)
    dfCY.to_csv(csvCY, index=False)

    # Plots
    plot_compare(dfN["t"], dfN["y_c"], "Newtonian",
                 dfCY["t"], dfCY["y_c"], "Carreau–Yasuda",
                 xlabel="t", ylabel="bubble centroid height y_c",
                 outpath=os.path.join(args.outdir, "compare_yc.png"))

    plot_compare(dfN["t"], dfN["AR"], "Newtonian",
                 dfCY["t"], dfCY["AR"], "Carreau–Yasuda",
                 xlabel="t", ylabel="aspect ratio AR = W/H",
                 outpath=os.path.join(args.outdir, "compare_ar.png"))

    plot_compare(dfN["t"], dfN["dyc_dt"], "Newtonian",
                 dfCY["t"], dfCY["dyc_dt"], "Carreau–Yasuda",
                 xlabel="t", ylabel="rise velocity dy_c/dt",
                 outpath=os.path.join(args.outdir, "compare_vel.png"))

    # Console summary
    print("\nWrote:")
    print("  ", csvN)
    print("  ", csvCY)
    print("  ", os.path.join(args.outdir, "compare_yc.png"))
    print("  ", os.path.join(args.outdir, "compare_ar.png"))
    print("  ", os.path.join(args.outdir, "compare_vel.png"))

    if missingN:
        print(f"\nNewtonian: missing {len(missingN)} VTU files referenced by PVD (first 10 shown):")
        for f in missingN[:10]:
            print("  ", f)

    if missingCY:
        print(f"\nCarreau–Yasuda: missing {len(missingCY)} VTU files referenced by PVD (first 10 shown):")
        for f in missingCY[:10]:
            print("  ", f)

    if len(dfN) == 0 or len(dfCY) == 0:
        print("\nWARNING: One of the datasets has zero readable timesteps. "
              "Make sure the .vtu files referenced in the .pvd are present.")

if __name__ == "__main__":
    main()