"""
============================================================================
 Monodomain vs Bidomain comparison for multi-vector low-energy stimulation
 2D ventricular-slab-with-vessels proof-of-concept
============================================================================

Purpose
-------
Answer the editor's / Reviewer 2's concern: does the direction-dependent
recruitment reported in the paper (using an ISOTROPIC MONODOMAIN with a
field-projection boundary condition) survive when the shock is instead
represented with a BIDOMAIN model that has UNEQUAL ANISOTROPY RATIOS -- the
mechanism (bulk virtual electrodes / sawtooth) that a monodomain cannot
produce?

The paper's claims are all RELATIVE (four directions beat one; different
directions recruit complementary sites). So we test whether the ORDERING
and the DIRECTION-DEPENDENCE are preserved, not absolute agreement.

What this script does
---------------------
  * builds a 2D slab (the ventricular-section analogue) with circular
    "vessels" as holes, using gmsh
  * assigns a transmural ROTATING fiber field (+60 deg -> -60 deg), which
    is what makes unequal anisotropy ratios matter
  * MONODOMAIN: reproduces the Cardiax boundary condition
        n . (sigma grad v) = A * (alpha/(1+alpha)) * sigma_o * (e . n)
    on field-exposed facets for t in (0, T_field], with the field
    direction rotating through the multi-vector sequence
  * BIDOMAIN (phi_e-coupled): the shock enters through the extracellular
    problem as a phi_e boundary flux; unequal intra/extra anisotropy ratios
    produce bulk polarization the monodomain cannot
  * ionics: TNNP-2006 (ported and validated in tnnp2006.py)
  * postprocessing: activation-time maps + global activation time tau,
    single- vs four-direction, at 0.5 and 0.2 V/cm, for both models

Runs on YOUR machine (needs dolfinx + gmsh). Not runnable in the assistant
environment. Tested components: the TNNP ODE port (see tnnp2006.py).

----------------------------------------------------------------------------
IMPORTANT KNOBS TO MATCH CARDIAX  (search "MATCH:")
----------------------------------------------------------------------------
  chi (surface-to-volume), Cm, timestep, sigma values, field coefficient.
  Defaults below use a standard cardiac convention (cm, ms, mV, mS/cm,
  uF/cm^2). The relative comparison is robust to the exact numbers, but for
  a like-for-like monodomain baseline set these to your .par/.xml values.
============================================================================
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import dolfinx
from dolfinx import fem, mesh as dmesh
import dolfinx.fem.petsc               # ensure submodule is loaded
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl

# gmsh<->dolfinx bridge. The module name varies across builds:
#   - dolfinx.io.gmshio   (common)
#   - dolfinx.io.gmsh     (this build: has gmsh.py)
# Fall back to reading a .msh via meshio only if neither exists.
_HAS_GMSHIO = True
gmshio = None
try:
    from dolfinx.io import gmshio
except Exception:
    try:
        from dolfinx.io import gmsh as gmshio   # singular name in some 0.11 builds
    except Exception:
        try:
            import dolfinx.io.gmshio as gmshio
        except Exception:
            _HAS_GMSHIO = False

import tnnp2006 as ion   # validated TNNP-2006 port


# ===========================================================================
# 0.  GLOBAL PARAMETERS  --  CARDIAX-NATIVE UNITS (um, ms, mS/um)
# ===========================================================================
#
# This is a FAITHFUL replica of the Cardiax monodomain (no magic gain).
# Everything is in Cardiax's own unit system so the constants are used
# VERBATIM from monodomain.cpp / cardiacproblem.cpp:
#
#   length     : micrometers (um)      <- meshes are in um
#   time       : milliseconds (ms)
#   conductivity: mS/um
#
# Assembly (exactly as Cardiax solve_parabolic):
#   A = M + theta * dtkappa * K
#   b = M v*  +  dtkappa * (boundary flux)
#   kappa   = (surface_to_volume / timestep) * um2_to_cm2
#   dtkappa = 1 / kappa
#   flux    = field_amplitude * (alpha/(1+alpha)) * sigma_o * (e . n)
#   field ON for t in (0, T_FIELD]; direction rotates on t in (0, 10].
#
comm = MPI.COMM_WORLD

# --- model parameters (clean PDE derivation, no Cardiax DTKAPPA) ---
# PDE:  beta * Cm * dv/dt = div(sigma grad v) - beta*I_ion
# theta-scheme groups the diffusion by  DIFF = dt/(beta*Cm).
# (Numerically DIFF equals Cardiax's old 1/[(surf2vol/dt)*1e-8], because their
#  um2_to_cm2=1e-8 factor was effectively the membrane capacitance Cm.)
BETA        = 0.14          # surface_to_volume [1/um] (Cardiax value)
CM          = 1.0e-8        # membrane capacitance 1 uF/cm^2 = 1e-8 uF/um^2
THETA       = 0.5           # theta_method (Crank-Nicolson)
DT          = 0.01          # timestep [ms]  (driver: -dt 0.05; TNNP RL hardcodes 0.05)

# ===========================================================================
# CONDUCTIVITIES  --  toggle isotropic  <->  Roth-based anisotropic
# ===========================================================================
# ANISOTROPY_ON = False : the original ISOTROPIC values (fiber direction has no
#                         effect, since sigma_l == sigma_t).
# ANISOTROPY_ON = True  : Roth-based transversely-isotropic values.
#     Monodomain: sig_mL/sig_mT = 6.25 (CV ratio 2.5), (sig_mL+sig_mT)/2 = sigma0.
#     Bidomain (unequal ratios): sig_i, sig_e chosen so the per-axis harmonic
#     means equal sig_mL, sig_mT exactly -> anisotropic mono and bido share the
#     SAME effective longitudinal/transverse conductivities and Roth ratios.
#     (intra anisotropy ratio 10:1, extra 2.5:1 -> unequal -> bulk virtual
#      electrodes, the mechanism the reviewer asked about.)
# The fiber DIRECTION is set separately by FIBER_MODE / FIBER_FIXED_DEG above.
ANISOTROPY_ON = True

# ---------------------------------------------------------------------------
# CASE SELECTOR for the 3-way comparison (mono-iso / mono-aniso / bido-aniso)
# ---------------------------------------------------------------------------
#   CASE = "A" : EQUAL-anisotropy ratio bidomain (validation).
#                Expect bido-aniso ~= mono-aniso (~1%); mono-iso differs.
#                This proves the solver is correct (bido reduces to mono).
#   CASE = "C" : UNEQUAL-anisotropy ratio bidomain (the real physics).
#                Expect bido-aniso != mono-aniso (genuine bidomain effect);
#                trustworthy BECAUSE case A validated the solver.
# Both cases use 45-degree fibers (oblique to the +x field). Outputs go to
# out_caseA/ or out_caseC/ to keep them organized.
CASE = "C"                 # "A" (equal, validation) or "C" (unequal, physics)
BIDO_VALIDATE = (CASE == "A")
OUTDIR = f"out_case{CASE}"

if ANISOTROPY_ON:
    # --- monodomain (anisotropic, Roth) [mS/um] ---
    SIGMA_MONO_L = 0.00034483
    SIGMA_MONO_T = 0.00005517
    if BIDO_VALIDATE:
        # CASE A: equal-anisotropy bidomain -> should reproduce mono-aniso
        SIGMA_I_L = 0.00068966
        SIGMA_I_T = 0.00011034
        SIGMA_E_L = 0.00068966
        SIGMA_E_T = 0.00011034
    else:
        # CASE C: unequal-anisotropy bidomain (Roth): intra 10:1, extra 2.5:1
        SIGMA_I_L = 0.00068966
        SIGMA_I_T = 0.00006897
        SIGMA_E_L = 0.00068966
        SIGMA_E_T = 0.00027586
else:
    # --- monodomain (original ISOTROPIC) [mS/um] ---
    SIGMA_MONO_L = 0.0002
    SIGMA_MONO_T = 0.0002
    # --- bidomain (isotropic, equal-ratio) [mS/um] ---
    SIGMA_I_L = 0.0014
    SIGMA_I_T = 0.0014
    SIGMA_E_L = 0.00023333
    SIGMA_E_T = 0.00023333

# --- field boundary coefficient (coeff_robin_vec), all PHYSICAL ---
ALPHA_FIELD = 6.0           # sigma_i = 6 sigma_e  -> alpha/(1+alpha) = 6/7
SIGMA_O     = 0.00067       # blood conductivity 0.67 S/m = 0.00067 mS/um
# field_amplitude values (the driver's -fa), in mV/um:
#   0.05 mV/um = 0.5 V/cm ,  0.01 mV/um = 0.1 V/cm  (x0.1 factor: 1 V/cm = 0.1 mV/um)
FIELD_AMP   = {"0.5": 0.05, "0.2": 0.02}   # label (V/cm) -> fa (mV/um)
T_FIELD     = 3.0           # field-on window [ms]  (final experiment: 3 ms pulse
                            #   to the arterial-tree boundary, 0.5 V/cm)
T_ROTATE    = 3.0           # single-direction here, so rotate window = pulse

# --- 2D dimensionality correction (NOT a free gain) ---
# surface_to_volume=0.14 and um2_to_cm2=1e-8 in dtkappa were derived for Cardiax's
# 3D meshes. In a 2D slab the boundary-facet-integral to element-integral ratio
# differs, so the same dtkappa over-drives the field (peakV overshoots the
# physiological TNNP AP peak of ~+40 mV; we observed ~+85 mV). This ONE scalar
# scales the FIELD FLUX ONLY (not the diffusion, so conduction velocity stays
# physical) and is set by the internal physiological anchor peakV -> ~+40 mV.
# It is applied IDENTICALLY to monodomain and bidomain, so it cannot bias the
# relative (mono-vs-bido, 1dir-vs-4dir) comparisons. Tune once via 'verify'.
GEOM_2D_FACTOR = 5.0 #0.5 #4.0 #4        # <-- set so verify gives peakV ~ +40 mV (see below)


# --- WHERE the field is applied (facet tags: 1=outer boundary, 2=vessel walls) ---
#   "vessels"  -> vessel walls only (tag 2)   [Fig.10 mechanism]
#   "boundary" -> outer boundary only (tag 1)
#   "both"     -> both (paper's "vessel + boundary" condition)
STIM_TARGET   = "vessels"

# Field-on duration for the recruitment (short-pulse) experiments.
T_FIELD_PULSE = 3.0             # [ms]; Fig.10 uses a 3 ms vessel pulse

# multi-vector sequences (paper: +x,-x,+y,-y for Ndir=4)
SEQ_1DIR = ["X"]
SEQ_4DIR = ["X", "x", "Y", "y"]

# --- time horizon ---
T_TOTAL = 30.0      # ms; extended so the corrected, slower conductivities fully activate
# Activation rule.
# Cardiax uses (v>0 OR v<-90) as a STOPPING heuristic. But for a physically
# meaningful ACTIVATION TIME we must count only genuine DEPOLARIZATION
# (an upstroke crossing V_ACT), not field-induced hyperpolarization -- otherwise
# a weak field that never fires the tissue produces a spurious small tau from a
# few hyperpolarized nodes. We track both:
#   depolarized(v): real activation (v crosses V_ACT upward)
#   touched(v):     the Cardiax heuristic, kept for reference/consistency
V_ACT = -20.0   # mV upstroke threshold for "activated" (well above rest -85)

def depolarized(v):
    return v > V_ACT

def touched(v):                      # Cardiax heuristic (reference only)
    return (v > 0.0) | (v < -90.0)

# A reported tau is only trustworthy if a real fraction of tissue depolarized.
MIN_ACT_FRAC = 0.90   # require >=90% depolarized for tau to be meaningful

# --- geometry (2D slab, MICROMETERS to match Cardiax mesh units) ---
# 1 cm = 1e4 um. Slab ~ 2 cm x 1 cm = 20000 x 10000 um.
SLAB_W = 20000.0    # width  [um]
SLAB_H = 10000.0    # height (transmural) [um]
VESSELS = [         # (cx, cy, radius) in um
    (6000.0, 5500.0, 1200.0),
    (12000.0, 4000.0, 1500.0),
    (15000.0, 7000.0, 1000.0),
]
MESH_H  = 200.0     # target element size [um] (~0.02 cm)

# ---------------------------------------------------------------------------
# FIBER ORIENTATION CONTROL
# ---------------------------------------------------------------------------
# The conductivity tensor is  sigma = sigma_t*I + (sigma_l - sigma_t) f (x) f ,
# with fiber direction f = (cos a, sin a). Anisotropy is present only when
# sigma_l != sigma_t (set those in the SIGMA_* blocks). This section controls
# the fiber DIRECTION field a(x):
#
#   FIBER_MODE = "rotating" : transmural linear rotation FIB_ENDO_DEG -> FIB_EPI_DEG
#                             (the original behaviour)
#   FIBER_MODE = "fixed"    : a single uniform fiber angle FIBER_FIXED_DEG
#                             everywhere (use 45 to test fibers NOT aligned with
#                             the x/y electric-field directions)
#
# To disable anisotropy entirely, keep sigma_l == sigma_t (then f is irrelevant).
FIBER_MODE      = "fixed"    # "rotating", "fixed", or "vessel"
# For the Case A/C comparison use "fixed" at 45 deg (fibres oblique to the +x
# field, uniform). ("vessel" curves fibres around holes -- for the VE sweep;
# "rotating" is transmural rotation.)
FIBER_FIXED_DEG = 45.0        # uniform fiber angle [deg] when FIBER_MODE=="fixed"

# fiber rotation (used when FIBER_MODE == "rotating"):
# angle varies linearly with y (transmural), +60 -> -60 deg
FIB_ENDO_DEG = +60.0
FIB_EPI_DEG  = -60.0

# --- diffusion time grouping (computed once) ---
#   DIFF = dt / (beta * Cm)   [replaces Cardiax dtkappa; numerically identical]
DIFF = DT / (BETA * CM)


# ===========================================================================
# 1.  MESH GENERATION  (gmsh -> dolfinx)
# ===========================================================================

def build_mesh():
    """Slab with circular vessel holes. Tags:
         boundary facet markers:
            1 = outer slab boundary (field-exposed)
            2 = vessel walls        (field-exposed)
       Everything on markers 1 and 2 is 'field-exposed' (Gamma_E).
    """
    gmsh.initialize()
    gmsh.model.add("slab")

    # outer rectangle
    slab = gmsh.model.occ.addRectangle(0, 0, 0, SLAB_W, SLAB_H)
    # vessel disks
    disks = []
    for (cx, cy, r) in VESSELS:
        disks.append((2, gmsh.model.occ.addDisk(cx, cy, 0, r, r)))
    # cut vessels out of slab
    out, _ = gmsh.model.occ.cut([(2, slab)], disks)
    gmsh.model.occ.synchronize()

    # tag the surface
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, tag=100)   # tissue

    # classify boundary curves: outer vs vessel by centroid distance
    curves = gmsh.model.getEntities(1)
    outer_ids, vessel_ids = [], []
    for (_, cid) in curves:
        com = gmsh.model.occ.getCenterOfMass(1, cid)
        x, y = com[0], com[1]
        on_outer = (abs(x) < 1e-6 or abs(x - SLAB_W) < 1e-6 or
                    abs(y) < 1e-6 or abs(y - SLAB_H) < 1e-6)
        (outer_ids if on_outer else vessel_ids).append(cid)
    gmsh.model.addPhysicalGroup(1, outer_ids, tag=1)   # outer boundary
    gmsh.model.addPhysicalGroup(1, vessel_ids, tag=2)  # vessel walls

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MESH_H)
    gmsh.model.mesh.generate(2)

    if _HAS_GMSHIO:
        # signature varies slightly across builds; try the common forms
        try:
            res = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
        except TypeError:
            res = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
        gmsh.finalize()
        # return type varies:
        #   older: tuple (mesh, cell_tags, facet_tags)
        #   newer: MeshData object with attributes (names differ)
        if isinstance(res, tuple):
            domain, cell_tags, facet_tags = res[0], res[1], res[2]
        else:
            domain = getattr(res, "mesh", None)
            cell_tags = (getattr(res, "cell_tags", None)
                         or getattr(res, "cell_meshtags", None))
            facet_tags = (getattr(res, "facet_tags", None)
                          or getattr(res, "facet_meshtags", None))
            if domain is None:
                raise RuntimeError(
                    "Unrecognized model_to_mesh return; attributes present: "
                    + str([a for a in dir(res) if not a.startswith('_')]))
    else:
        # Fallback: write a .msh (v2.2, which dolfinx reads reliably) and read it.
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write("slab.msh")
        gmsh.finalize()
        domain, cell_tags, facet_tags = read_msh_fallback("slab.msh")

    if facet_tags is None:
        raise RuntimeError(
            "facet_tags came back None -- gmsh physical groups for boundaries "
            "(tags 1 and 2) were not transferred. Check the addPhysicalGroup calls.")
    return domain, cell_tags, facet_tags


def read_msh_fallback(path):
    """Read a gmsh .msh (v2.2) into dolfinx without the gmshio module,
    using meshio to translate to a form dolfinx.io can ingest.
    Requires the 'meshio' package (pip install meshio)."""
    import meshio
    m = meshio.read(path)

    # cells
    tri = None; line = None
    tri_tags = None; line_tags = None
    for cb in m.cells:
        if cb.type == "triangle":
            tri = cb.data
        elif cb.type == "line":
            line = cb.data
    # physical tags stored in cell_data 'gmsh:physical'
    phys = m.cell_data_dict.get("gmsh:physical", {})
    tri_tags = phys.get("triangle", None)
    line_tags = phys.get("line", None)

    points = m.points[:, :2]  # 2D

    # Build a dolfinx mesh with a P1 triangle coordinate element.
    from basix.ufl import element as bx_element
    gdim = 2
    c_el = bx_element("Lagrange", "triangle", 1, shape=(gdim,))
    domain = dolfinx.mesh.create_mesh(
        comm, tri.astype(np.int64), points.astype(np.float64),
        ufl.Mesh(c_el))

    # cell tags
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    if tri_tags is not None:
        ncells = tri.shape[0]
        cell_tags = dolfinx.mesh.meshtags(
            domain, tdim, np.arange(ncells, dtype=np.int32),
            tri_tags.astype(np.int32))
    else:
        cell_tags = None

    # facet tags: map the 'line' cells (endpoints) to local facet indices
    domain.topology.create_connectivity(tdim - 1, 0)
    domain.topology.create_connectivity(tdim - 1, tdim)
    facet_tags = None
    if line is not None and line_tags is not None:
        # build a lookup from sorted vertex-pair -> tag
        want = {tuple(sorted(map(int, e))): int(t)
                for e, t in zip(line, line_tags)}
        f2v = domain.topology.connectivity(tdim - 1, 0)
        nfac = domain.topology.index_map(tdim - 1).size_local
        fidx, fval = [], []
        for f in range(nfac):
            verts = tuple(sorted(f2v.links(f).tolist()))
            if verts in want:
                fidx.append(f); fval.append(want[verts])
        if fidx:
            order = np.argsort(fidx)
            facet_tags = dolfinx.mesh.meshtags(
                domain, tdim - 1,
                np.array(fidx, dtype=np.int32)[order],
                np.array(fval, dtype=np.int32)[order])
    return domain, cell_tags, facet_tags


# ===========================================================================
# 2.  FIBER FIELD + CONDUCTIVITY TENSORS
# ===========================================================================

def fiber_angle(x):
    """Fiber angle a(x) [radians].
       FIBER_MODE == "fixed"    -> uniform FIBER_FIXED_DEG everywhere.
       FIBER_MODE == "rotating" -> linear transmural rotation with y.
       FIBER_MODE == "vessel"   -> potential-flow fibers curving AROUND each
                                   vessel (Connolly, Vigmond & Bishop 2017,
                                   PLoS ONE, Eq.6: fibres = streamlines of
                                   potential flow around a cylinder). This is
                                   what generates the first-order (fibre-
                                   curvature) virtual electrode at vessels."""
    if FIBER_MODE == "fixed":
        return np.deg2rad(FIBER_FIXED_DEG)
    if FIBER_MODE == "rotating":
        frac = x[1] / SLAB_H
        deg = FIB_ENDO_DEG + (FIB_EPI_DEG - FIB_ENDO_DEG) * frac
        return np.deg2rad(deg)
    if FIBER_MODE == "vessel":
        # Far-field fibre direction (uniform) at angle phi0:
        phi0 = np.deg2rad(FIBER_FIXED_DEG)
        U = np.array([np.cos(phi0), np.sin(phi0)])   # unit far-field "flow"
        # Superpose the potential-flow velocity perturbation of each vessel.
        # For a cylinder radius a at center c, uniform flow U, the 2D potential
        # -flow velocity is:
        #   v = U + (a^2/r^2) * [ 2 (U.rhat) rhat - U ]   (doublet term),
        # written in the vessel-centred frame; rhat = (dx,dy)/r.
        v = U.astype(float).copy()
        px, py = x[0], x[1]
        for (cx, cy, a) in VESSELS:
            dx, dy = px - cx, py - cy
            r2 = dx*dx + dy*dy
            if r2 < (a*a):        # inside the hole (shouldn't happen); skip
                continue
            rhat = np.array([dx, dy]) / np.sqrt(r2)
            Udotr = U[0]*rhat[0] + U[1]*rhat[1]
            v += (a*a / r2) * (2.0*Udotr*rhat - U)
        return np.arctan2(v[1], v[0])
    # default
    return np.deg2rad(FIBER_FIXED_DEG)

def conductivity_tensor(domain, sigma_l, sigma_t):
    """Build a DG0 tensor field  sigma = sigma_t I + (sigma_l - sigma_t) f f^T
       with the rotating fiber f = (cos a, sin a)."""
    V_ten = fem.functionspace(domain, ("DG", 0, (2, 2)))
    sig = fem.Function(V_ten)

    # evaluate at cell midpoints (version-robust)
    tdim = domain.topology.dim
    ncells = domain.topology.index_map(tdim).size_local
    ents = np.arange(ncells, dtype=np.int32)
    try:
        midpoints = dolfinx.mesh.compute_midpoints(domain, tdim, ents)
    except AttributeError:
        # fallback: average the vertex coordinates of each cell manually
        domain.topology.create_connectivity(tdim, 0)
        c2v = domain.topology.connectivity(tdim, 0)
        xg = domain.geometry.x
        # geometry dofmap maps cells -> geometry nodes
        gdofs = domain.geometry.dofmap
        midpoints = np.zeros((ncells, 3))
        for c in range(ncells):
            verts = gdofs[c]
            midpoints[c, :] = xg[verts].mean(axis=0)

    vals = np.zeros((ncells, 4))
    for c in range(ncells):
        a = fiber_angle(midpoints[c])
        f = np.array([np.cos(a), np.sin(a)])
        M = sigma_t*np.eye(2) + (sigma_l - sigma_t)*np.outer(f, f)
        vals[c, :] = M.reshape(-1)
    sig.x.array[:] = vals.reshape(-1)
    return sig


# ===========================================================================
# 3.  FIELD DIRECTION HELPERS  (matches Cardiax set_field_direction)
# ===========================================================================

def dir_vec(ch):
    return {
        "X": ( 1.0, 0.0), "x": (-1.0, 0.0),
        "Y": ( 0.0, 1.0), "y": ( 0.0,-1.0),
    }[ch]

def current_direction(t, seq, t_field=T_FIELD, t_rotate=T_ROTATE):
    """Cardiax-faithful field timing.
    Field is ON for t in (0, t_field]; returns None outside that window.
    Direction changes happen on t in (0, t_rotate] every t_rotate/len(seq) ms
    (Cardiax uses change_inc = 10/field_changes with the change loop gated by
    t<=10, while the flux itself is gated by t<=12).
    For the short recruitment pulse, pass t_field=t_rotate=T_FIELD_PULSE.
    """
    if t <= 0 or t > t_field:
        return None
    ndir = len(seq)
    # direction index advances within the rotation window, then holds the last
    seg = t_rotate / ndir
    idx = min(int(t // seg), ndir - 1) if t <= t_rotate else ndir - 1
    return dir_vec(seq[idx])

def stim_ds(domain, facet_tags):
    """Return the ds measure restricted to the facets selected by STIM_TARGET.
       1 = outer boundary, 2 = vessel walls."""
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    if STIM_TARGET == "vessels":
        return ds(2)
    elif STIM_TARGET == "boundary":
        return ds(1)
    elif STIM_TARGET == "both":
        return ds(1) + ds(2)
    else:
        raise ValueError(f"STIM_TARGET must be vessels/boundary/both, got {STIM_TARGET}")


# ===========================================================================
# 4.  MONODOMAIN SOLVER
# ===========================================================================

def run_monodomain(domain, facet_tags, sigma_fun, field_amp, seq, label,
                   record_every=0, snapshot_at=None):
    """Operator-split monodomain with the Cardiax field boundary condition.
       Returns (activation_time_array, tau, frac, reliable, snapshot).
       If record_every>0, also writes a Vm time series to vm_mono_<label>.xdmf.
       If snapshot_at is not None, 'snapshot' is the Vm array captured at the
       first step with t >= snapshot_at (else None). Use snapshot_at=T_FIELD_PULSE
       to capture the END-OF-SHOCK polarization (virtual electrodes)."""
    V = fem.functionspace(domain, ("Lagrange", 1))
    imap = V.dofmap.index_map
    ndof = imap.size_local                       # owned dofs (serial: all)
    nloc = imap.size_local + imap.num_ghosts     # local array length

    v_n  = fem.Function(V); v_n.name = "Vm"   # transmembrane potential at step n
    v_np = fem.Function(V)
    snapshot = None                            # end-of-shock Vm, if requested

    # --- optional Vm time-series writer ---
    xdmf = None
    if record_every > 0:
        _ensure_outdir()
        xdmf = dolfinx.io.XDMFFile(comm, f"{OUTDIR}/vm_mono_{label}.xdmf", "w")
        xdmf.write_mesh(domain)

    # --- ODE state (TNNP) held on the P1 dofs (owned only) ---
    y = ion.init_state(ndof, "EPI")
    v_n.x.array[:ndof] = y[0, :]
    v_n.x.scatter_forward()

    # --- variational forms (Cardiax-faithful) ---
    u  = ufl.TrialFunction(V)
    w  = ufl.TestFunction(V)
    dx = ufl.dx

    # System matrix EXACTLY as Cardiax: A = M + theta * dtkappa * K
    #   M = int u w dx ,  K = int (sigma grad u).(grad w) dx
    a_form = fem.form((u*w)*dx
                      + (THETA*DIFF)*ufl.dot(sigma_fun*ufl.grad(u), ufl.grad(w))*dx)
    A = assemble_matrix(a_form)
    A.assemble()

    # RHS mass part: Cardiax uses b = Mi v0 with Mi = M - theta*dtkappa*K
    # (Crank-Nicolson). Reuse v_n as the coefficient (v* after the ODE step).
    m_form = fem.form((v_n*w)*dx
                      - ((1.0-THETA)*DIFF)*ufl.dot(sigma_fun*ufl.grad(v_n), ufl.grad(w))*dx)

    # field boundary form (direction via Constants, reused each step)
    # Cardiax: flux coeff = fa*(alpha/(1+alpha))*sigma_o*(e.n); added as
    #          f.add(node, belvec * dtkappa).  So we scale the facet integral
    #          by DIFF here.
    n = ufl.FacetNormal(domain)
    e_x = fem.Constant(domain, PETSc.ScalarType(0.0))
    e_y = fem.Constant(domain, PETSc.ScalarType(0.0))
    amp = fem.Constant(domain, PETSc.ScalarType(0.0))
    # MONODOMAIN boundary forcing (per 2D comparison doc):  A = a * sigma_o / 2.
    # The 1/2 is alpha/(1+alpha) with alpha=1 (Roth equal-anisotropy counterpart
    # used for the monodomain representation). Sign: +A*(e.n) (table Sec.6:
    # right wall, e.n=+1 -> +a*sigma_o/2).
    coeff = GEOM_2D_FACTOR * amp * 0.5 * SIGMA_O    # A = a*sigma_o/2
    e_dot_n = e_x*n[0] + e_y*n[1]
    ds_stim = stim_ds(domain, facet_tags)   # respects STIM_TARGET
    L_field_form = fem.form(DIFF * coeff * e_dot_n * w * ds_stim)

    # preallocated RHS vector (matches operator A exactly, version-proof)
    b = A.createVecRight()

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("jacobi")
    solver.setTolerances(rtol=1e-8)

    act_time = np.full(ndof, np.nan)
    activated = np.zeros(ndof, dtype=bool)   # depolarization-based
    vmax_seen = -1e9

    nsteps = int(T_TOTAL/DT)
    for k in range(nsteps):
        t = (k+1)*DT

        # (1) reaction (TNNP); field enters via PDE boundary condition, not here
        y = ion.step(y, 0.0, DT, "EPI", use_sac=False)
        v_n.x.array[:ndof] = y[0, :]
        v_n.x.scatter_forward()

        # (2) RHS = M v*  (+ field flux while field is on)
        with b.localForm() as bloc:
            bloc.set(0.0)
        assemble_vector(b, m_form)

        d = current_direction(t, seq)
        if d is not None:
            e_x.value, e_y.value = d
            amp.value = field_amp
            assemble_vector(b, L_field_form)     # adds into b

        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # (3) diffusion solve
        solver.solve(b, v_np.x.petsc_vec)
        v_np.x.scatter_forward()

        # (4) copy PDE solution back into ODE state (v is state var 0)
        y[0, :] = v_np.x.array[:ndof]
        v_n.x.array[:ndof] = v_np.x.array[:ndof]
        v_n.x.scatter_forward()

        # (5) activation bookkeeping (owned dofs) -- DEPOLARIZATION only
        vv = v_np.x.array[:ndof]
        vmax_seen = max(vmax_seen, float(vv.max()))
        newly = depolarized(vv) & (~activated)
        act_time[newly] = t
        activated |= newly

        # (6) end-of-shock snapshot (virtual-electrode polarization pattern)
        if snapshot_at is not None and snapshot is None and t >= snapshot_at:
            snapshot = vv.copy()

        # (7) optional Vm snapshot (guard against HDF5 time-series quirks)
        if xdmf is not None and (k % record_every == 0):
            try:
                xdmf.write_function(v_n, t)
            except Exception as ex:
                if comm.rank == 0:
                    print(f"  [warn] Vm time-series write disabled ({type(ex).__name__}); "
                          f"acttime map is unaffected.")
                try: xdmf.close()
                except Exception: pass
                xdmf = None

        if activated.all():
            break

    if xdmf is not None:
        try:
            xdmf.write_function(v_n, (k+1)*DT)   # final frame
        except Exception:
            pass
        xdmf.close()

    frac = activated.sum()/ndof
    reliable = frac >= MIN_ACT_FRAC
    tau = np.nanmax(act_time) if reliable else float('nan')
    if comm.rank == 0:
        flag = "" if reliable else "  <-- UNRELIABLE (tissue did not activate)"
        taus = f"{tau:.2f}" if reliable else "  n/a"
        print(f"[MONO {label}] fa={field_amp} mV/um (={10*field_amp:g} V/cm) seq={''.join(seq)}  "
              f"tau={taus} ms  activated={100*frac:.1f}%  "
              f"peakV={vmax_seen:.1f} mV{flag}")
    return act_time, tau, frac, reliable, snapshot


# ===========================================================================
# 5.  BIDOMAIN SOLVER (phi_e-coupled)
# ===========================================================================

def run_bidomain(domain, facet_tags, field_amp, seq, label, record_every=0,
                 snapshot_at=None, sigmas=None):
    """
    Parabolic-elliptic bidomain, operator split.
      Parabolic (v):  chi Cm dv/dt = div(sigma_i grad v) + div(sigma_i grad phi_e) - chi I_ion
      Elliptic (phi_e): div((sigma_i+sigma_e) grad phi_e) = -div(sigma_i grad v)
    Shock enters as a phi_e Neumann flux on Gamma_E (full a*sigma_o, opposite
    sign to the monodomain -- see the 2D comparison document).
    Unequal anisotropy ratios (sigma_i vs sigma_e) generate bulk virtual
    electrodes absent from the monodomain.
    Returns (activation_time_array, tau, frac, reliable, snapshot).
    If record_every>0, writes Vm and phi_e time series. If snapshot_at is set,
    'snapshot' is the Vm array captured at the first step with t >= snapshot_at
    (end-of-shock polarization = virtual electrodes)."""
    V  = fem.functionspace(domain, ("Lagrange", 1))
    imap = V.dofmap.index_map
    ndof = imap.size_local

    v_n   = fem.Function(V); v_n.name = "Vm"
    v_np  = fem.Function(V)
    phi_e = fem.Function(V); phi_e.name = "phi_e"
    snapshot = None

    # optional Vm and phi_e time-series writers (separate files: XDMF is happier
    # writing one field per file across a time series)
    xdmf = None
    xdmf_phi = None
    if record_every > 0:
        _ensure_outdir()
        xdmf = dolfinx.io.XDMFFile(comm, f"{OUTDIR}/vm_bido_{label}.xdmf", "w")
        xdmf.write_mesh(domain)
        xdmf_phi = dolfinx.io.XDMFFile(comm, f"{OUTDIR}/phie_bido_{label}.xdmf", "w")
        xdmf_phi.write_mesh(domain)

    # intra/extra tensors with rotating fibers.
    # sigmas override (dict iL,iT,eL,eT) lets a caller run several bidomain
    # configs in one session (e.g. the 'final' mode) without touching globals.
    if sigmas is None:
        siL, siT, seL, seT = SIGMA_I_L, SIGMA_I_T, SIGMA_E_L, SIGMA_E_T
    else:
        siL, siT, seL, seT = sigmas["iL"], sigmas["iT"], sigmas["eL"], sigmas["eT"]
    sig_i = conductivity_tensor(domain, siL, siT)
    sig_e = conductivity_tensor(domain, seL, seT)

    y = ion.init_state(ndof, "EPI")
    v_n.x.array[:ndof] = y[0, :]
    v_n.x.scatter_forward()

    u  = ufl.TrialFunction(V)
    w  = ufl.TestFunction(V)
    dx = ufl.dx
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n  = ufl.FacetNormal(domain)

    # ----- elliptic operator for phi_e:  A_e phi_e = -Ki v + field -----
    a_e_form = fem.form(ufl.dot((sig_i+sig_e)*ufl.grad(u), ufl.grad(w))*dx)
    A_e = assemble_matrix(a_e_form)
    A_e.assemble()

    # pure-Neumann elliptic -> constant nullspace
    ns = PETSc.NullSpace().create(constant=True, comm=comm)
    A_e.setNullSpace(ns)
    A_e.setNearNullSpace(ns)   # helps gamg

    # field flux form for phi_e (direction via Constants).
    # BIDOMAIN extracellular boundary flux (per 2D comparison doc):
    #   n.(sigma_e grad phi_e) = -a * sigma_o * (e.n)   -- FULL a*sigma_o, NO 1/2.
    # The 1/2 that appears in the monodomain case is the alpha/(1+alpha)=1/2
    # reduction; the bidomain carries the complete extracellular flux a*sigma_o.
    # The MINUS sign matches the doc's boundary table (Sec.6): left wall e.n=-1
    # -> +a*sigma_o on phi_e, i.e. flux = -a*sigma_o*(e.n). This is also why the
    # membrane depolarizes the SAME wall as the monodomain (mono uses +A(e.n) on
    # v; the opposite phi_e sign gives the same depolarization side).
    e_x = fem.Constant(domain, PETSc.ScalarType(0.0))
    e_y = fem.Constant(domain, PETSc.ScalarType(0.0))
    amp = fem.Constant(domain, PETSc.ScalarType(0.0))
    coeff = GEOM_2D_FACTOR * amp * SIGMA_O           # FULL a*sigma_o (no 1/2)
    e_dot_n = e_x*n[0] + e_y*n[1]
    ds_stim = stim_ds(domain, facet_tags)   # respects STIM_TARGET
    L_field_form = fem.form(-coeff*e_dot_n*w*ds_stim)

    # coupling term  -int sigma_i grad v . grad w   (v via v_couple coefficient)
    v_couple = fem.Function(V)
    L_couple_form = fem.form(-ufl.dot(sig_i*ufl.grad(v_couple), ufl.grad(w))*dx)

    ksp_e = PETSc.KSP().create(comm)
    ksp_e.setOperators(A_e)
    ksp_e.setType("cg"); ksp_e.getPC().setType("gamg")
    ksp_e.setTolerances(rtol=1e-8)

    # ----- parabolic operator for v (Cardiax-faithful: M + theta*dtkappa*Ki) -----
    a_p_form = fem.form((u*w)*dx
                        + (THETA*DIFF)*ufl.dot(sig_i*ufl.grad(u), ufl.grad(w))*dx)
    A_p = assemble_matrix(a_p_form)
    A_p.assemble()
    ksp_p = PETSc.KSP().create(comm)
    ksp_p.setOperators(A_p)
    ksp_p.setType("cg"); ksp_p.getPC().setType("jacobi")
    ksp_p.setTolerances(rtol=1e-8)

    # parabolic RHS pieces (Crank-Nicolson mass) and the phi_e coupling source
    m_form   = fem.form((v_n*w)*dx
                        - ((1.0-THETA)*DIFF)*ufl.dot(sig_i*ufl.grad(v_n), ufl.grad(w))*dx)
    phi_src  = fem.Function(V)
    L_phi_form = fem.form(-DIFF*ufl.dot(sig_i*ufl.grad(phi_src), ufl.grad(w))*dx)

    # preallocated vectors (match operators exactly, version-proof)
    b_e = A_e.createVecRight()   # elliptic RHS
    b_p = A_p.createVecRight()   # parabolic RHS

    act_time = np.full(ndof, np.nan)
    activated = np.zeros(ndof, dtype=bool)
    vmax_seen = -1e9

    nsteps = int(T_TOTAL/DT)
    for k in range(nsteps):
        t = (k+1)*DT

        # (1) reaction (TNNP)
        y = ion.step(y, 0.0, DT, "EPI", use_sac=False)
        v_star = y[0, :].copy()
        v_n.x.array[:ndof] = v_star
        v_n.x.scatter_forward()

        # (2) elliptic solve for phi_e given v* and the field
        v_couple.x.array[:ndof] = v_star
        v_couple.x.scatter_forward()
        with b_e.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b_e, L_couple_form)
        d = current_direction(t, seq)
        if d is not None:
            e_x.value, e_y.value = d
            amp.value = field_amp
            assemble_vector(b_e, L_field_form)     # adds into b_e
        b_e.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ns.remove(b_e)                             # project RHS onto range
        ksp_e.solve(b_e, phi_e.x.petsc_vec)
        phi_e.x.scatter_forward()

        # (3) parabolic solve for v
        phi_src.x.array[:ndof] = phi_e.x.array[:ndof]
        phi_src.x.scatter_forward()
        with b_p.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b_p, m_form)
        assemble_vector(b_p, L_phi_form)           # adds into b_p
        b_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ksp_p.solve(b_p, v_np.x.petsc_vec)
        v_np.x.scatter_forward()

        # (4) copy back into ODE state
        y[0, :] = v_np.x.array[:ndof]
        v_n.x.array[:ndof] = v_np.x.array[:ndof]
        v_n.x.scatter_forward()

        # (5) activation (DEPOLARIZATION only)
        vv = v_np.x.array[:ndof]
        vmax_seen = max(vmax_seen, float(vv.max()))
        newly = depolarized(vv) & (~activated)
        act_time[newly] = t
        activated |= newly

        # (6) end-of-shock snapshot (virtual-electrode polarization)
        if snapshot_at is not None and snapshot is None and t >= snapshot_at:
            snapshot = vv.copy()

        # (7) optional Vm and phi_e snapshot (guard against HDF5 quirks)
        if xdmf is not None and (k % record_every == 0):
            try:
                xdmf.write_function(v_n, t)
                xdmf_phi.write_function(phi_e, t)
            except Exception as ex:
                if comm.rank == 0:
                    print(f"  [warn] Vm/phi_e time-series write disabled "
                          f"({type(ex).__name__}); acttime map is unaffected.")
                for xf in (xdmf, xdmf_phi):
                    try: xf.close()
                    except Exception: pass
                xdmf = None; xdmf_phi = None

        if activated.all():
            break

    if xdmf is not None:
        try:
            xdmf.write_function(v_n, (k+1)*DT)
            xdmf_phi.write_function(phi_e, (k+1)*DT)
        except Exception:
            pass
        xdmf.close(); xdmf_phi.close()

    frac = activated.sum()/ndof
    reliable = frac >= MIN_ACT_FRAC
    tau = np.nanmax(act_time) if reliable else float('nan')
    if comm.rank == 0:
        flag = "" if reliable else "  <-- UNRELIABLE (tissue did not activate)"
        taus = f"{tau:.2f}" if reliable else "  n/a"
        print(f"[BIDO {label}] fa={field_amp} mV/um (={10*field_amp:g} V/cm) seq={''.join(seq)}  "
              f"tau={taus} ms  activated={100*frac:.1f}%  "
              f"peakV={vmax_seen:.1f} mV{flag}")
    return act_time, tau, frac, reliable, snapshot


# ===========================================================================
# 6.  DRIVER
# ===========================================================================

def _ensure_outdir():
    import os
    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
    comm.Barrier()

def save_field(domain, arr, name):
    _ensure_outdir()
    V = fem.functionspace(domain, ("Lagrange", 1))
    f = fem.Function(V); f.name = name
    ndof = V.dofmap.index_map.size_local
    f.x.array[:ndof] = np.nan_to_num(arr[:ndof], nan=-1.0)
    f.x.scatter_forward()
    with dolfinx.io.XDMFFile(comm, f"{OUTDIR}/{name}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(f)

def verify(domain, facet_tags):
    """Acceptance test for the FAITHFUL replica (no gain to tune).
    Runs monodomain, fa=0.05 mV/um (=0.5 V/cm), +x, field on both boundary and
    vessels (STIM_TARGET), and reports tau + peakV. If the replica is correct,
    the tissue should ACTIVATE at 0.5 V/cm (as Cardiax does) with a physiological
    peakV ~ +40 mV and a tau in the tens-of-ms range comparable to Cardiax.

    Usage:  python multivector_bidomain_2d.py verify
    """
    sig_mono = conductivity_tensor(domain, SIGMA_MONO_L, SIGMA_MONO_T)
    if comm.rank == 0:
        print("\n=== FAITHFUL-REPLICA VERIFICATION ===")
        print(f"fa = {FIELD_AMP['0.5']} mV/um = 0.5 V/cm ,  +x , "
              f"STIM_TARGET={STIM_TARGET}")
        print(f"DIFF = {DIFF:.4e} , theta = {THETA} , dt = {DT} ms")
        print("Expect: tissue ACTIVATES, peakV ~ +40 mV, tau in tens of ms.\n")
    run_monodomain(domain, facet_tags, sig_mono, FIELD_AMP["0.5"], SEQ_1DIR,
                   "verify", record_every=25)


def run_recruitment(domain, facet_tags, model="mono"):
    """Reproduce the paper's Fig.10-style experiment: apply a SHORT pulse to the
    arterial-tree interface in each single direction separately, and record the
    activation-time map. Different directions recruit different vessel regions.
    Writes acttime_recruit_<model>_<dir>.xdmf for X,x,Y,y and a combined
    'earliest activation across directions' map. Open all in ParaView.

    This is the direct 2D analogue of Fig.10 (a-e) and is the figure that
    answers the reviewer for BOTH models when run with model='mono' and 'bido'.
    """
    sig_mono = conductivity_tensor(domain, SIGMA_MONO_L, SIGMA_MONO_T)
    # (direction char for the solver, unambiguous name for filenames)
    dirs = [("X", "posX"), ("x", "negX"), ("Y", "posY"), ("y", "negY")]
    maps = {}
    for dch, dname in dirs:
        lab = f"recruit_{model}_{dname}"
        if model == "mono":
            at, tau, fr, ok, _ = run_monodomain(domain, facet_tags, sig_mono,
                                             FIELD_AMP["0.5"], [dch], lab, record_every=25)
        else:
            at, tau, fr, ok, _ = run_bidomain(domain, facet_tags,
                                           FIELD_AMP["0.5"], [dch], lab, record_every=25)
        save_field(domain, at, f"acttime_{lab}")
        maps[dname] = at
        if comm.rank == 0:
            frac = 100.0*np.mean(~np.isnan(at))
            print(f"  [{model} recruit {dname}] recruited {frac:.1f}% of tissue")

    # combined: earliest activation time across the four directions
    stack = np.vstack([np.nan_to_num(maps[dn], nan=np.inf) for _, dn in dirs])
    combined = np.min(stack, axis=0)
    combined[np.isinf(combined)] = np.nan
    save_field(domain, combined, f"acttime_recruit_{model}_combined")
    if comm.rank == 0:
        union = 100.0*np.mean(~np.isnan(combined))
        print(f"  [{model} recruit] UNION of all 4 directions recruited "
              f"{union:.1f}% of tissue (vs each single direction alone)")


def run_sweep(domain, facet_tags):
    """Field-strength sweep comparing THREE cases at each field:
        1. monodomain ISOTROPIC   (sigma = sigma0 = 0.002, option (a): the mean
                                    (sig_mL+sig_mT)/2, so same effective scale)
        2. monodomain ANISOTROPIC (Roth: sig_mL=0.00034483, sig_mT=0.00005517)
        3. bidomain   ANISOTROPIC (Roth unequal: uses module SIGMA_I/E globals)
    Single direction (+x). GEOM_2D_FACTOR is held FIXED; only the field varies.
    Fields swept (V/cm): SWEEP_VCM. Internally fa[mV/um] = 0.1 * (V/cm).

    Usage:  python multivector_bidomain_2d.py sweep
    """
    SWEEP_VCM = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0] #[0.5, 5.0]            # V/cm: final visualization experiment (0.5 V/cm, 3 ms vessel pulse)
    SEQ = SEQ_1DIR                                # +x only

    # case conductivity tensors (built once)
    sig_iso   = conductivity_tensor(domain, 0.002, 0.002)              # case 1
    sig_aniso = conductivity_tensor(domain, SIGMA_MONO_L, SIGMA_MONO_T) # case 2 (Roth mono)
    if comm.rank == 0:
        print(f"\n=== FIELD SWEEP (CASE {CASE}): mono-iso vs mono-aniso vs bido-aniso ===")
        if CASE == "A":
            print("  CASE A: EQUAL-ratio bidomain (validation) -> expect")
            print("          bido-aniso ~= mono-aniso (~1%); mono-iso differs.")
        else:
            print("  CASE C: UNEQUAL-ratio bidomain (physics) -> expect")
            print("          bido-aniso != mono-aniso (genuine bidomain effect).")
        print(f"  outputs -> {OUTDIR}/ ,  fibers=45deg fixed")
        print(f"  GEOM_2D_FACTOR={GEOM_2D_FACTOR} (fixed), dt={DT}, +x, "
              f"STIM_TARGET={STIM_TARGET}")
        print(f"  case2 mono-aniso: sig_mL={SIGMA_MONO_L}, sig_mT={SIGMA_MONO_T}")
        print(f"  case3 bido:       sig_iL={SIGMA_I_L}, sig_iT={SIGMA_I_T}, "
              f"sig_eL={SIGMA_E_L}, sig_eT={SIGMA_E_T}")
        print(f"  End-of-shock Vm snapshot captured at t={T_FIELD_PULSE} ms "
              f"(virtual-electrode polarization).\n")

    rows = []
    # Vm (and phi_e for bidomain) time series: write a frame every SWEEP_REC
    # steps. With DT=0.05 and a short shock this is cheap; lower it for finer
    # temporal resolution. Set to 0 to disable the time series (snapshot only).
    SWEEP_REC = 10 #2
    for vcm in SWEEP_VCM:    
        fa = 0.1 * vcm            # V/cm -> mV/um
        tag = f"{vcm:g}Vcm"
        snap_t = T_FIELD_PULSE    # capture Vm right at end of the shock

        at1, t1, f1, ok1, s1 = run_monodomain(domain, facet_tags, sig_iso, fa, SEQ,
                                    f"sweep_monoiso_{tag}", record_every=SWEEP_REC,
                                    snapshot_at=snap_t)
        save_field(domain, at1, f"acttime_sweep_monoiso_{tag}")
        if s1 is not None: save_field(domain, s1, f"vmShock_sweep_monoiso_{tag}")

        at2, t2, f2, ok2, s2 = run_monodomain(domain, facet_tags, sig_aniso, fa, SEQ,
                                    f"sweep_monoaniso_{tag}", record_every=SWEEP_REC,
                                    snapshot_at=snap_t)
        save_field(domain, at2, f"acttime_sweep_monoaniso_{tag}")
        if s2 is not None: save_field(domain, s2, f"vmShock_sweep_monoaniso_{tag}")

        at3, t3, f3, ok3, s3 = run_bidomain(domain, facet_tags, fa, SEQ,
                                    f"sweep_bidoaniso_{tag}", record_every=SWEEP_REC,
                                    snapshot_at=snap_t)
        save_field(domain, at3, f"acttime_sweep_bidoaniso_{tag}")
        if s3 is not None: save_field(domain, s3, f"vmShock_sweep_bidoaniso_{tag}")

        rows.append((vcm, t1, f1, ok1, t2, f2, ok2, t3, f3, ok3))

    if comm.rank == 0:
        def cell(tau, frac, ok):
            return (f"{tau:6.2f}/{100*frac:3.0f}%" if ok
                    else f"  n/a/{100*frac:3.0f}%")
        print("\n==================== SWEEP SUMMARY ====================")
        print("tau[ms]/activated%   (n/a = <90% activated, tau unreliable)")
        print(f"{'E[V/cm]':>8} | {'mono-iso':>13} | {'mono-aniso':>13} | "
              f"{'bido-aniso':>13}")
        print("-"*60)
        for (vcm, t1,f1,ok1, t2,f2,ok2, t3,f3,ok3) in rows:
            print(f"{vcm:>8g} | {cell(t1,f1,ok1):>13} | {cell(t2,f2,ok2):>13} | "
                  f"{cell(t3,f3,ok3):>13}")
        print("\nActivation-time maps written as acttime_sweep_*_<E>Vcm.xdmf "
              "(one per case per field).")


# def run_final(domain, facet_tags):
#     """FINAL comparison: run all FOUR cases in one session, under IDENTICAL
#     conditions (same mesh, field, timing, GEOM_2D_FACTOR, fibres), so the panel
#     is a controlled comparison. Conductivities are fixed here explicitly and do
#     NOT depend on the CASE / ANISOTROPY_ON flags.

#         1. mono-iso        : isotropic monodomain, sigma = 0.002 (mean of L,T)
#         2. mono-aniso      : Roth anisotropic monodomain (sig_mL, sig_mT)
#         3. bido-equal      : equal-ratio bidomain  (reduces to mono-aniso; ~1%)
#         4. bido-unequal    : unequal-ratio Roth bidomain (the physics)

#     All bidomain conductivities share the SAME per-axis harmonic means as
#     mono-aniso, so cases 2,3,4 have matched effective L/T conductivities and
#     differ only in the intra/extra anisotropy split.

#     Field: 0.5 V/cm, +x, applied for T_FIELD ms to STIM_TARGET; simulate to
#     T_TOTAL. Vm (and phi_e) time series + end-of-shock snapshot + activation map
#     are written to out_final/ for each case.

#     Usage:  python multivector_bidomain_2d.py final
#     """
#     global OUTDIR
#     OUTDIR = "out_final"

#     # --- fixed conductivities (independent of CASE flag) [mS/um] ---
#     MONO_ISO   = (0.0002, 0.0002)
#     MONO_ANISO = (0.00034483, 0.00005517)
#     BIDO_EQUAL   = dict(iL=0.00068966, iT=0.00011034, eL=0.00068966, eT=0.00011034)
#     BIDO_UNEQUAL = dict(iL=0.00068966, iT=0.00006897, eL=0.00068966, eT=0.00027586)

#     fa = FIELD_AMP["0.5"]          # 0.5 V/cm
#     REC = 4                        # Vm/phi_e frame every REC steps

#     if comm.rank == 0:
#         print("\n=== FINAL 4-CASE COMPARISON (identical conditions) ===")
#         print(f"  field 0.5 V/cm (+x), T_FIELD={T_FIELD} ms, T_TOTAL={T_TOTAL} ms,")
#         print(f"  GEOM_2D_FACTOR={GEOM_2D_FACTOR}, dt={DT}, FIBER_MODE={FIBER_MODE}, "
#               f"STIM_TARGET={STIM_TARGET}")
#         print(f"  outputs -> {OUTDIR}/\n")

#     results = {}

#     sig1 = conductivity_tensor(domain, *MONO_ISO)
#     at, tau, fr, ok, snap = run_monodomain(domain, facet_tags, sig1, fa, SEQ_1DIR,
#                                            "final_mono_iso", record_every=REC,
#                                            snapshot_at=T_FIELD)
#     save_field(domain, at, "acttime_final_mono_iso")
#     if snap is not None: save_field(domain, snap, "vmShock_final_mono_iso")
#     results["mono-iso"] = (tau, fr, ok)

#     sig2 = conductivity_tensor(domain, *MONO_ANISO)
#     at, tau, fr, ok, snap = run_monodomain(domain, facet_tags, sig2, fa, SEQ_1DIR,
#                                            "final_mono_aniso", record_every=REC,
#                                            snapshot_at=T_FIELD)
#     save_field(domain, at, "acttime_final_mono_aniso")
#     if snap is not None: save_field(domain, snap, "vmShock_final_mono_aniso")
#     results["mono-aniso"] = (tau, fr, ok)

#     at, tau, fr, ok, snap = run_bidomain(domain, facet_tags, fa, SEQ_1DIR,
#                                          "final_bido_equal", record_every=REC,
#                                          snapshot_at=T_FIELD, sigmas=BIDO_EQUAL)
#     save_field(domain, at, "acttime_final_bido_equal")
#     if snap is not None: save_field(domain, snap, "vmShock_final_bido_equal")
#     results["bido-equal"] = (tau, fr, ok)

#     at, tau, fr, ok, snap = run_bidomain(domain, facet_tags, fa, SEQ_1DIR,
#                                          "final_bido_unequal", record_every=REC,
#                                          snapshot_at=T_FIELD, sigmas=BIDO_UNEQUAL)
#     save_field(domain, at, "acttime_final_bido_unequal")
#     if snap is not None: save_field(domain, snap, "vmShock_final_bido_unequal")
#     results["bido-unequal"] = (tau, fr, ok)

#     if comm.rank == 0:
#         print("\n==================== FINAL SUMMARY ====================")
#         print(f"{'case':>14} | {'tau[ms]':>9} | {'activated%':>10}")
#         print("-"*42)
#         for k, (tau, fr, ok) in results.items():
#             taus = f"{tau:.2f}" if ok else "  n/a"
#             print(f"{k:>14} | {taus:>9} | {100*fr:>9.1f}%")
#         print(f"\nPer-case outputs in {OUTDIR}/:")
#         print("  vmShock_final_<case>.xdmf   (Vm at end of shock, t=T_FIELD)")
#         print("  vm_{mono,bido}_final_<case>.xdmf  (Vm time series)")
#         print("  phie_bido_final_<case>.xdmf (phi_e time series, bidomain)")
#         print("  acttime_final_<case>.xdmf   (activation-time map)")
#         print("\nNote: mono-aniso and bido-equal should closely match (~1%);")
#         print("bido-unequal is the physics test (no monodomain equivalent).")

def run_final(domain, facet_tags):
    """FINAL comparison across a field sweep: run all FOUR cases at each field
    intensity, under IDENTICAL conditions (same mesh, timing, GEOM_2D_FACTOR,
    fibres, STIM_TARGET). Conductivities are fixed here explicitly and do NOT
    depend on the CASE / ANISOTROPY_ON flags.

        1. mono-iso     : isotropic monodomain, sigma = 0.0002 mS/um
        2. mono-aniso   : Roth anisotropic monodomain (sig_mL, sig_mT)
        3. bido-equal   : equal-ratio bidomain (reduces to mono-aniso; ~1%)
        4. bido-unequal : unequal-ratio Roth bidomain (the physics)

    All bidomain conductivities share the SAME per-axis harmonic means as
    mono-aniso, so cases 2,3,4 have matched effective L/T conductivities and
    differ only in the intra/extra anisotropy split.

    Fields swept (V/cm): FINAL_VCM. Field applied for T_FIELD ms to STIM_TARGET,
    +x, simulate to T_TOTAL. Filenames are tagged with the field, e.g.
    vmShock_final_mono_iso_0.5Vcm.xdmf, so intensities don't overwrite.

    Usage:  python multivector_bidomain_2d.py final
    """
    global OUTDIR
    OUTDIR = "out_final"

    FINAL_VCM = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0]   # V/cm to sweep

    # --- fixed conductivities (independent of CASE flag) [mS/um] ---
    MONO_ISO     = (0.0002, 0.0002)
    MONO_ANISO   = (0.00034483, 0.00005517)
    BIDO_EQUAL   = dict(iL=0.00068966, iT=0.00011034, eL=0.00068966, eT=0.00011034)
    BIDO_UNEQUAL = dict(iL=0.00068966, iT=0.00006897, eL=0.00068966, eT=0.00027586)

    REC = 4                        # Vm/phi_e frame every REC steps

    # build the conductivity tensors once (independent of field)
    sig_iso   = conductivity_tensor(domain, *MONO_ISO)
    sig_aniso = conductivity_tensor(domain, *MONO_ANISO)

    if comm.rank == 0:
        print("\n=== FINAL 4-CASE COMPARISON, FIELD SWEEP ===")
        print(f"  fields (V/cm): {FINAL_VCM}")
        print(f"  +x, T_FIELD={T_FIELD} ms, T_TOTAL={T_TOTAL} ms, "
              f"GEOM_2D_FACTOR={GEOM_2D_FACTOR}, dt={DT},")
        print(f"  FIBER_MODE={FIBER_MODE}, STIM_TARGET={STIM_TARGET}")
        print(f"  outputs -> {OUTDIR}/  (filenames tagged with field)\n")

    results = {}   # (case, vcm) -> (tau, frac, ok)

    for vcm in FINAL_VCM:
        fa = 0.1 * vcm            # V/cm -> mV/um
        tag = f"{vcm:g}Vcm"
        if comm.rank == 0:
            print(f"\n--- field {vcm:g} V/cm (fa={fa:g} mV/um) ---")

        # 1. mono-iso
        at, tau, fr, ok, snap = run_monodomain(
            domain, facet_tags, sig_iso, fa, SEQ_1DIR,
            f"final_mono_iso_{tag}", record_every=REC, snapshot_at=T_FIELD)
        save_field(domain, at, f"acttime_final_mono_iso_{tag}")
        if snap is not None: save_field(domain, snap, f"vmShock_final_mono_iso_{tag}")
        results[("mono-iso", vcm)] = (tau, fr, ok)

        # 2. mono-aniso
        at, tau, fr, ok, snap = run_monodomain(
            domain, facet_tags, sig_aniso, fa, SEQ_1DIR,
            f"final_mono_aniso_{tag}", record_every=REC, snapshot_at=T_FIELD)
        save_field(domain, at, f"acttime_final_mono_aniso_{tag}")
        if snap is not None: save_field(domain, snap, f"vmShock_final_mono_aniso_{tag}")
        results[("mono-aniso", vcm)] = (tau, fr, ok)

        # 3. bido-equal
        at, tau, fr, ok, snap = run_bidomain(
            domain, facet_tags, fa, SEQ_1DIR,
            f"final_bido_equal_{tag}", record_every=REC, snapshot_at=T_FIELD,
            sigmas=BIDO_EQUAL)
        save_field(domain, at, f"acttime_final_bido_equal_{tag}")
        if snap is not None: save_field(domain, snap, f"vmShock_final_bido_equal_{tag}")
        results[("bido-equal", vcm)] = (tau, fr, ok)

        # 4. bido-unequal
        at, tau, fr, ok, snap = run_bidomain(
            domain, facet_tags, fa, SEQ_1DIR,
            f"final_bido_unequal_{tag}", record_every=REC, snapshot_at=T_FIELD,
            sigmas=BIDO_UNEQUAL)
        save_field(domain, at, f"acttime_final_bido_unequal_{tag}")
        if snap is not None: save_field(domain, snap, f"vmShock_final_bido_unequal_{tag}")
        results[("bido-unequal", vcm)] = (tau, fr, ok)

    if comm.rank == 0:
        print("\n==================== FINAL SUMMARY ====================")
        print("tau[ms] / activated%   (n/a = <90% activated)")
        cases = ["mono-iso", "mono-aniso", "bido-equal", "bido-unequal"]
        header = f"{'E[V/cm]':>8} | " + " | ".join(f"{c:>16}" for c in cases)
        print(header); print("-"*len(header))
        for vcm in FINAL_VCM:
            cells = []
            for c in cases:
                tau, fr, ok = results[(c, vcm)]
                cells.append(f"{tau:6.2f}/{100*fr:3.0f}%" if ok
                             else f"   n/a/{100*fr:3.0f}%")
            print(f"{vcm:>8g} | " + " | ".join(f"{x:>16}" for x in cells))
        print(f"\nOutputs in {OUTDIR}/ , filenames tagged _<E>Vcm.")
        print("mono-aniso and bido-equal should match (~1%); bido-unequal is the physics.")

def main():
    import sys
    domain, cell_tags, facet_tags = build_mesh()

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify(domain, facet_tags)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "final":
        run_final(domain, facet_tags)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        run_sweep(domain, facet_tags)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "recruit":
        # Fig.10-style per-direction vessel recruitment, both models.
        if comm.rank == 0:
            print(f"\n=== RECRUITMENT (Fig.10-style)  STIM_TARGET={STIM_TARGET}, "
                  f"pulse={T_FIELD_PULSE} ms ===")
        run_recruitment(domain, facet_tags, "mono")
        run_recruitment(domain, facet_tags, "bido")
        return

    # record_every: write a Vm frame every N steps for ParaView (0 = off).
    # With DT=0.02, record_every=25 -> a frame every 0.5 ms.
    REC = 25

    sig_mono = conductivity_tensor(domain, SIGMA_MONO_L, SIGMA_MONO_T)

    results = {}
    for estr, eamp in FIELD_AMP.items():
        for seq, sname in [(SEQ_1DIR, "1dir"), (SEQ_4DIR, "4dir")]:
            lab = f"{estr}_{sname}"

            at_m, tau_m, fr_m, ok_m, _ = run_monodomain(
                domain, facet_tags, sig_mono, eamp, seq, lab, record_every=REC)
            save_field(domain, at_m, f"acttime_mono_{lab}")

            at_b, tau_b, fr_b, ok_b, _ = run_bidomain(
                domain, facet_tags, eamp, seq, lab, record_every=REC)
            save_field(domain, at_b, f"acttime_bido_{lab}")

            results[lab] = dict(tau_mono=tau_m, frac_mono=fr_m, ok_mono=ok_m,
                                tau_bido=tau_b, frac_bido=fr_b, ok_bido=ok_b)

    # ---- summary: does the ordering survive?  (reliability-aware) ----
    if comm.rank == 0:
        def fmt(tau, ok):
            return f"{tau:8.2f}" if ok else "   n/a  "
        print("\n==================== SUMMARY ====================")
        print(f"{'case':>10} | {'tau_mono':>9} ({'%act':>5}) | "
              f"{'tau_bido':>9} ({'%act':>5})")
        print("-"*56)
        for lab, r in results.items():
            print(f"{lab:>10} | {fmt(r['tau_mono'],r['ok_mono'])} "
                  f"({100*r['frac_mono']:4.0f}%) | "
                  f"{fmt(r['tau_bido'],r['ok_bido'])} "
                  f"({100*r['frac_bido']:4.0f}%)")
        print("\nKey checks (only meaningful when BOTH cases activated >=90%):")
        for estr in FIELD_AMP:
            r1 = results[f"{estr}_1dir"]; r4 = results[f"{estr}_4dir"]
            # monodomain
            if r1["ok_mono"] and r4["ok_mono"]:
                m = f"4dir<1dir = {r4['tau_mono']<r1['tau_mono']} " \
                    f"({r1['tau_mono']:.1f}->{r4['tau_mono']:.1f} ms)"
            else:
                m = "n/a (a case did not activate; see %act)"
            # bidomain
            if r1["ok_bido"] and r4["ok_bido"]:
                b = f"4dir<1dir = {r4['tau_bido']<r1['tau_bido']} " \
                    f"({r1['tau_bido']:.1f}->{r4['tau_bido']:.1f} ms)"
            else:
                b = "n/a (a case did not activate; see %act)"
            print(f"  E={estr}:")
            print(f"      mono: {m}")
            print(f"      bido: {b}")
        print("\nVm time series written to vm_{mono,bido}_<case>.xdmf")
        print("Activation-time maps written to acttime_{mono,bido}_<case>.xdmf")
        print("Open either in ParaView. For Vm, use the time slider; for")
        print("acttime, color by the field (value = ms; -1 = never activated).")

if __name__ == "__main__":
    main()
