"""
Publication figure: activation-time maps, monodomain vs bidomain
(meshio + matplotlib tripcolor -- no PyVista/VTK)

Reads the activation-time maps written by `multivector_bidomain_2d.py final`
(out_final/acttime_final_*.xdmf) or the corrected archived NPZ maps and renders
the four cases in a 2 x 2 grid. Activation time is not transient, so there is
one map per case.

Cases:
    mono_iso, mono_aniso, bido_equal, bido_unequal

Non-activated nodes were saved as -1 (sentinel); they are masked (shown white).

Usage:
    python plot_acttime.py --cases mono_iso mono_aniso bido_unequal
    python plot_acttime.py --cases all --out figure_acttime.png
    python plot_acttime.py --cases all --cmap viridis --tmax 9

Requires: meshio, numpy, matplotlib.
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

INDIR = "out_final"
CASES = {
    "mono_iso":     ("acttime_final_mono_iso.xdmf",     "isotropic mono."),
    "mono_aniso":   ("acttime_final_mono_aniso.xdmf",   "anisotropic mono."),
    "bido_equal":   ("acttime_final_bido_equal.xdmf",   "anisotropic bido. (equal)"),
    "bido_unequal": ("acttime_final_bido_unequal.xdmf", "anisotropic bido. (unequal)"),
}
NPZ_CASES = {
    "mono_iso":     "acttime_mono-iso_0.5Vcm.npz",
    "mono_aniso":   "acttime_mono-aniso_0.5Vcm.npz",
    "bido_equal":   "acttime_bido-equal_0.5Vcm.npz",
    "bido_unequal": "acttime_bido-unequal_0.5Vcm.npz",
}
SENTINEL = -1.0   # non-activated nodes


def use_times_font():
    for name in ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                 "FreeSerif", "DejaVu Serif"]:
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = [name]
            return name
    plt.rcParams["font.family"] = "serif"
    return "serif(default)"


def read_map(xdmf_path):
    """Read a dolfinx single-field XDMF. dolfinx writes a (single-step) TEMPORAL
    grid, which meshio's plain reader rejects ('only one grid'); the
    TimeSeriesReader handles it. Falls back to reading the paired .h5 via h5py."""
    import meshio
    # 1) meshio TimeSeriesReader (dolfinx writes a 1-step temporal collection)
    try:
        with meshio.xdmf.TimeSeriesReader(xdmf_path) as reader:
            pts, cells = reader.read_points_cells()
            tris = None
            for cb in cells:
                if cb.type == "triangle":
                    tris = cb.data
            nsteps = reader.num_steps
            _, pd, _ = reader.read_data(nsteps - 1)
            key = list(pd.keys())[0]
            val = np.asarray(pd[key]).ravel()
        if tris is None:
            raise RuntimeError("no triangles via TimeSeriesReader")
        return pts[:, :2], tris, val
    except Exception as e1:
        print(f"  [info] TimeSeriesReader failed ({e1}); trying h5py fallback")

    # 2) h5py fallback: read the paired .h5 directly (dolfinx layout)
    import h5py
    h5path = xdmf_path[:-5] + ".h5" if xdmf_path.endswith(".xdmf") else xdmf_path + ".h5"
    with h5py.File(h5path, "r") as f:
        # geometry + topology live under Mesh/<name>/{geometry,topology}
        geo = topo = None
        def _find(name, obj):
            nonlocal geo, topo
            if name.endswith("geometry") and geo is None:
                geo = obj[()]
            if name.endswith("topology") and topo is None:
                topo = obj[()]
        f.visititems(lambda n, o: _find(n, o) if isinstance(o, h5py.Dataset) else None)
        pts = np.asarray(geo)[:, :2]
        tris = np.asarray(topo).reshape(-1, 3)
        # the function values: first dataset under a group containing the field
        val = None
        def _findval(name, obj):
            nonlocal val
            if isinstance(obj, h5py.Dataset) and ("Function" in name or "values" in name.lower()):
                arr = np.asarray(obj[()]).ravel()
                if arr.size == pts.shape[0] and val is None:
                    val = arr
        f.visititems(_findval)
        if val is None:
            raise RuntimeError(f"could not locate field values in {h5path}")
    return pts, tris, val


def read_npz_map(npz_path):
    """Read the corrected archived activation-map format."""
    with np.load(npz_path) as data:
        pts = np.asarray(data["points"])[:, :2]
        tris = np.asarray(data["triangles"], dtype=int)
        val = np.asarray(data["activation"]).ravel()
    return pts, tris, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["all"])
    ap.add_argument("--indir", default=INDIR,
                    help="directory containing the XDMF or archived NPZ maps")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=None,
                    help="max activation time [ms] for colour scale (auto if unset)")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--out", default="figure_acttime.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fontname = use_times_font()
    print(f"[font] using {fontname}")

    case_keys = list(CASES.keys()) if args.cases == ["all"] else args.cases
    for k in case_keys:
        if k not in CASES:
            raise SystemExit(f"unknown case '{k}'. choose from {list(CASES.keys())}")

    maps = {}
    gmax = 0.0
    for k in case_keys:
        path = os.path.join(args.indir, CASES[k][0])
        if os.path.exists(path):
            pts, tris, val = read_map(path)
        else:
            npz_path = os.path.join(args.indir, NPZ_CASES[k])
            if not os.path.exists(npz_path):
                raise SystemExit(
                    f"missing files: {path} and {npz_path} "
                    "(run 'final' mode first)"
                )
            pts, tris, val = read_npz_map(npz_path)
        maps[k] = (pts, tris, val)
        active = val[val > SENTINEL + 0.5]   # exclude sentinel
        vmax_k = active.max() if active.size else 0.0
        gmax = max(gmax, vmax_k)
        print(f"[read] {k}: activation max = {vmax_k:.2f} ms")

    tmax = args.tmax if args.tmax is not None else gmax
    norm = Normalize(vmin=args.tmin, vmax=tmax)

    ncase = len(case_keys)
    # ---- 2x2 grid layout ----
    nrow, ncol = 2, 2
    fig_w = 2.6*ncol + 0.9
    fig_h = 2.4*nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), squeeze=False)

    for i, k in enumerate(case_keys):
        ax = axes.flat[i]        # fills TL, TR, BL, BR in reading order
        pts, tris, val = maps[k]
        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        # mask triangles touching a non-activated (sentinel) node
        mask = np.any(val[tris] <= SENTINEL + 0.5, axis=1)
        triang.set_mask(mask)
        ax.tripcolor(triang, val, shading="gouraud", cmap=args.cmap,
                     norm=norm, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(CASES[k][1], fontsize=10, pad=6)

    # hide any unused panels (if fewer than nrow*ncol cases)
    for j in range(ncase, nrow*ncol):
        axes.flat[j].set_visible(False)

    fig.subplots_adjust(left=0.03, right=0.88, top=0.93, bottom=0.03,
                        wspace=0.05, hspace=0.02)
    sm = ScalarMappable(norm=norm, cmap=args.cmap); sm.set_array([])

    bar_h = 0.6                      # colorbar height (fraction of figure)
    bar_y = 0.5 - bar_h / 2.0        # centers it vertically
    cax = fig.add_axes([0.90, bar_y, 0.015, bar_h])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("activation time (ms)", fontsize=10)
    cb.ax.tick_params(labelsize=10)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"[saved] {args.out}  ({nrow}x{ncol} grid, tmax={tmax:.2f} ms, dpi={args.dpi})")


if __name__ == "__main__":
    main()
