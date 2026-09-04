"""Corrected four-case monodomain/bidomain comparison.

This is a serial scikit-fem execution backend for the variational forms in
``multivector_bidomain_2d.py``.  It is used only because DOLFINx is unavailable
in the execution image; the P1 finite-element mesh, Crank--Nicolson split,
TNNP-2006 reaction step, field boundary integrals, geometry, time step and
activation definition are the same as in the supplied script.

The conductivity values are the corrected Cardiax-native values in mS/um.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, FacetBasis, LinearForm, MeshTri, asm
from skfem.element import ElementTriP1
from skfem.helpers import dot, grad

import tnnp2006 as ion


HERE = Path(__file__).resolve().parent
OUT = HERE / "out_corrected"
OUT.mkdir(exist_ok=True)

# Geometry and protocol (identical to the supplied final-mode experiment).
SLAB_W, SLAB_H = 20000.0, 10000.0  # um
VESSELS = [
    (6000.0, 5500.0, 1200.0),
    (12000.0, 4000.0, 1500.0),
    (15000.0, 7000.0, 1000.0),
]
MESH_H = 200.0
FIBRE_DEG = 45.0
DT = 0.01
T_TOTAL = 30.0  # extended from 15 ms so corrected, slower cases fully activate
T_FIELD = 3.0
THETA = 0.5
BETA = 0.14
CM = 1.0e-8
DIFF = DT / (BETA * CM)
GEOM_2D_FACTOR = 5.0
SIGMA_O = 0.00067
V_ACT = -20.0
MIN_ACT_FRAC = 0.90
FIELDS = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0]

# Corrected Cardiax-native conductivities [mS/um].
COND = {
    "mono-iso": dict(m=(0.0002, 0.0002)),
    "mono-aniso": dict(m=(0.00034483, 0.00005517)),
    "bido-equal": dict(
        i=(0.00068966, 0.00011034),
        e=(0.00068966, 0.00011034),
    ),
    "bido-unequal": dict(
        i=(0.00068966, 0.00006897),
        e=(0.00068966, 0.00027586),
    ),
}

LABELS = {
    "mono-iso": "isotropic mono.",
    "mono-aniso": "anisotropic mono.",
    "bido-equal": "anisotropic bido. (equal)",
    "bido-unequal": "anisotropic bido. (unequal)",
}


def make_mesh() -> tuple[MeshTri, np.ndarray]:
    """Generate the original slab-with-holes mesh and return vessel facets."""
    msh_path = OUT / "slab_corrected.msh"
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("slab_corrected")
    slab = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, SLAB_W, SLAB_H)
    disks = [(2, gmsh.model.occ.addDisk(cx, cy, 0.0, r, r))
             for cx, cy, r in VESSELS]
    gmsh.model.occ.cut([(2, slab)], disks)
    gmsh.model.occ.synchronize()
    surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfaces, 100)
    outer, vessel = [], []
    for _, cid in gmsh.model.getEntities(1):
        x, y, _ = gmsh.model.occ.getCenterOfMass(1, cid)
        on_outer = (abs(x) < 1e-6 or abs(x - SLAB_W) < 1e-6 or
                    abs(y) < 1e-6 or abs(y - SLAB_H) < 1e-6)
        (outer if on_outer else vessel).append(cid)
    gmsh.model.addPhysicalGroup(1, outer, 1)
    gmsh.model.addPhysicalGroup(1, vessel, 2)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MESH_H)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    raw = meshio.read(msh_path)
    tri_blocks = [c.data for c in raw.cells if c.type == "triangle"]
    line_blocks = [c.data for c in raw.cells if c.type == "line"]
    tris = np.vstack(tri_blocks)
    lines = np.vstack(line_blocks)
    line_tags = np.asarray(raw.cell_data_dict["gmsh:physical"]["line"])
    vessel_lines = lines[line_tags == 2]

    mesh = MeshTri(raw.points[:, :2].T, tris.T)
    facet_lookup = {
        tuple(sorted((int(mesh.facets[0, j]), int(mesh.facets[1, j])))): j
        for j in range(mesh.facets.shape[1])
    }
    vessel_facets = np.array([
        facet_lookup[tuple(sorted((int(a), int(b))))]
        for a, b in vessel_lines
    ], dtype=np.int32)
    return mesh, vessel_facets


def tensor(sig_l: float, sig_t: float) -> np.ndarray:
    a = np.deg2rad(FIBRE_DEG)
    f = np.array([np.cos(a), np.sin(a)])
    return sig_t * np.eye(2) + (sig_l - sig_t) * np.outer(f, f)


def stiffness(basis: Basis, dmat: np.ndarray):
    @BilinearForm
    def form(u, v, w):
        gu, gv = grad(u), grad(v)
        return (dmat[0, 0] * gu[0] * gv[0]
                + dmat[0, 1] * gu[1] * gv[0]
                + dmat[1, 0] * gu[0] * gv[1]
                + dmat[1, 1] * gu[1] * gv[1])
    return asm(form, basis).tocsr()


def vessel_normal_x(basis: Basis, vessel_facets: np.ndarray) -> np.ndarray:
    fb = FacetBasis(basis.mesh, ElementTriP1(), facets=vessel_facets)

    @LinearForm
    def form(v, w):
        return w.n[0] * v

    return np.asarray(asm(form, fb))


@BilinearForm
def mass_form(u, v, w):
    return u * v


@dataclass
class MonoSolver:
    name: str
    mass: object
    stiff: object
    field_vec: np.ndarray

    def __post_init__(self):
        self.left = (self.mass + THETA * DIFF * self.stiff).tocsc()
        self.right = (self.mass - (1.0 - THETA) * DIFF * self.stiff).tocsr()
        self.lu = splu(self.left)

    def run(self, vcm: float):
        ndof = self.mass.shape[0]
        y = ion.init_state(ndof, "EPI")
        act = np.full(ndof, np.nan)
        activated = np.zeros(ndof, dtype=bool)
        vmax = -np.inf
        fa = 0.1 * vcm
        load = DIFF * GEOM_2D_FACTOR * fa * 0.5 * SIGMA_O * self.field_vec
        for k in range(int(T_TOTAL / DT)):
            t = (k + 1) * DT
            y = ion.step(y, 0.0, DT, "EPI", use_sac=False)
            rhs = self.right @ y[0]
            if t <= T_FIELD:
                rhs = rhs + load
            v = self.lu.solve(rhs)
            y[0] = v
            vmax = max(vmax, float(v.max()))
            newly = (v > V_ACT) & (~activated)
            act[newly] = t
            activated |= newly
            if activated.all():
                break
        frac = float(activated.mean())
        ok = frac >= MIN_ACT_FRAC
        tau = float(np.nanmax(act)) if ok else np.nan
        return act, tau, frac, vmax


@dataclass
class BidoSolver:
    name: str
    mass: object
    ki: object
    ke: object
    field_vec: np.ndarray

    def __post_init__(self):
        self.ksum = (self.ki + self.ke).tocsr()
        self.lu_e = splu(self.ksum[1:, 1:].tocsc())  # phi_e[0]=0 gauge
        self.left = (self.mass + THETA * DIFF * self.ki).tocsc()
        self.right = (self.mass - (1.0 - THETA) * DIFF * self.ki).tocsr()
        self.lu_p = splu(self.left)

    def run(self, vcm: float):
        ndof = self.mass.shape[0]
        y = ion.init_state(ndof, "EPI")
        act = np.full(ndof, np.nan)
        activated = np.zeros(ndof, dtype=bool)
        vmax = -np.inf
        fa = 0.1 * vcm
        # Bidomain extracellular flux: -a sigma_o (e.n), with no 1/2.
        load = -GEOM_2D_FACTOR * fa * SIGMA_O * self.field_vec
        for k in range(int(T_TOTAL / DT)):
            t = (k + 1) * DT
            y = ion.step(y, 0.0, DT, "EPI", use_sac=False)
            vstar = y[0]
            rhs_e = -(self.ki @ vstar)
            if t <= T_FIELD:
                rhs_e = rhs_e + load
            # Constant projection reproduces the pure-Neumann nullspace removal.
            rhs_e = rhs_e - rhs_e.mean()
            phi = np.zeros(ndof)
            phi[1:] = self.lu_e.solve(rhs_e[1:])
            rhs_p = self.right @ vstar - DIFF * (self.ki @ phi)
            v = self.lu_p.solve(rhs_p)
            y[0] = v
            vmax = max(vmax, float(v.max()))
            newly = (v > V_ACT) & (~activated)
            act[newly] = t
            activated |= newly
            if activated.all():
                break
        frac = float(activated.mean())
        ok = frac >= MIN_ACT_FRAC
        tau = float(np.nanmax(act)) if ok else np.nan
        return act, tau, frac, vmax


def save_map(mesh: MeshTri, case: str, act: np.ndarray):
    np.savez_compressed(
        OUT / f"acttime_{case}_0.5Vcm.npz",
        points=mesh.p.T,
        triangles=mesh.t.T,
        activation=np.nan_to_num(act, nan=-1.0),
    )


def plot_maps(mesh: MeshTri, maps: dict[str, np.ndarray]):
    order = list(LABELS)
    valid = np.concatenate([v[np.isfinite(v)] for v in maps.values()])
    vmax = float(valid.max())
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.85))
    norm = matplotlib.colors.Normalize(0.0, vmax)
    for ax, case in zip(axes, order):
        val = maps[case]
        tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
        inactive_tri = np.any(~np.isfinite(val[mesh.t.T]), axis=1)
        # Light gray denotes tissue not activated within the observation window;
        # the circular cavities remain white because no tissue triangles exist.
        ax.tripcolor(tri, facecolors=np.zeros(mesh.nelements), shading="flat",
                     cmap=matplotlib.colors.ListedColormap(["#d9d9d9"]),
                     vmin=0.0, vmax=1.0, rasterized=True)
        tri.set_mask(inactive_tri)
        ax.tripcolor(tri, np.nan_to_num(val, nan=0.0), shading="gouraud",
                     cmap="turbo", norm=norm, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(LABELS[case], fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.91, top=0.88, bottom=0.05, wspace=0.10)
    cax = fig.add_axes([0.93, 0.18, 0.014, 0.64])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="turbo")
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("activation time (ms)", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_activation_maps_corrected.{suffix}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tau(rows: list[dict]):
    styles = {
        "mono-iso": dict(color="#4477AA", marker="o", ls="-"),
        "mono-aniso": dict(color="#EE6677", marker="s", ls="-"),
        "bido-equal": dict(color="#228833", marker="^", ls="--"),
        "bido-unequal": dict(color="#CCBB44", marker="D", ls="-"),
    }
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for case in LABELS:
        rr = [r for r in rows if r["case"] == case and np.isfinite(r["tau_ms"])]
        ax.plot([r["field_Vcm"] for r in rr], [r["tau_ms"] for r in rr],
                label=LABELS[case], markerfacecolor="white", linewidth=1.7,
                markersize=6.5, markeredgewidth=1.4, **styles[case])
    ax.set_xlabel(r"field amplitude $E$ (V/cm)", fontsize=12)
    ax.set_ylabel(r"activation time $\tau$ (ms)", fontsize=12)
    ax.tick_params(labelsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_tau_vs_E_corrected.{suffix}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    mesh, vessel_facets = make_mesh()
    basis = Basis(mesh, ElementTriP1())
    mass = asm(mass_form, basis).tocsr()
    field_vec = vessel_normal_x(basis, vessel_facets)
    print(f"mesh: {basis.N} nodes, {mesh.nelements} triangles, "
          f"{len(vessel_facets)} vessel facets", flush=True)

    solvers = {}
    for case in ("mono-iso", "mono-aniso"):
        sm = COND[case]["m"]
        solvers[case] = MonoSolver(case, mass, stiffness(basis, tensor(*sm)), field_vec)
    for case in ("bido-equal", "bido-unequal"):
        si, se = COND[case]["i"], COND[case]["e"]
        solvers[case] = BidoSolver(
            case, mass, stiffness(basis, tensor(*si)),
            stiffness(basis, tensor(*se)), field_vec)

    rows = []
    maps = {}
    for vcm in FIELDS:
        for case, solver in solvers.items():
            start = time.time()
            act, tau, frac, vmax = solver.run(vcm)
            row = dict(case=case, field_Vcm=vcm, tau_ms=tau,
                       activated_fraction=frac, peak_Vm_mV=vmax)
            rows.append(row)
            ts = "n/a" if not np.isfinite(tau) else f"{tau:.2f}"
            print(f"{case:14s} E={vcm:3.1f} V/cm  tau={ts:>5s} ms  "
                  f"activated={100*frac:5.1f}% peak={vmax:6.1f} mV  "
                  f"({time.time()-start:.1f}s)", flush=True)
            if abs(vcm - 0.5) < 1e-12:
                maps[case] = act.copy()
                save_map(mesh, case, act)

    with (OUT / "corrected_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot_maps(mesh, maps)
    plot_tau(rows)
    print(f"completed in {(time.time()-t0)/60:.1f} min; outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
