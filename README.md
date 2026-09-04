# Corrected 2-D monodomain--bidomain comparison

This directory contains the final code and archived data used to generate the
two new figures prepared for the revision of *Multi-Vector Low-Energy Cardiac
Stimulation Recruits Direction-Dependent Vascular and Boundary Hotspots*.

## Definitive execution files

- `run_corrected_skfem.py` is the serial P1 finite-element program that was
  actually executed to generate the corrected field sweep, the four
  activation-time maps at 0.5 V/cm, and the activation-time-versus-field
  figure.
- `tnnp2006.py` implements the ten Tusscher--Panfilov 2006 ventricular ionic
  model used by the execution driver.
- `plot_acttime.py` reads the archived activation maps and generated the final
  2 x 2 activation-map figure.
- `multivector_bidomain_2d_corrected.py` is the corresponding DOLFINx version.
  It is included as the corrected reference implementation, but the supplied
  numerical results were generated with the scikit-fem driver because DOLFINx
  was unavailable in the execution environment.

The older standalone `plot_tau_vs_E.py` is deliberately not included because
it contained hard-coded results from an earlier, superseded run.

## Reproduce the simulations and both figures

Install the serial-backend dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete corrected sweep:

```bash
python run_corrected_skfem.py
```

This writes the results to `out_corrected/`, including
`corrected_results.csv`, the four compressed activation maps, the mesh, and
`figure_tau_vs_E_corrected.{pdf,png}`. It also writes the earlier horizontal
four-panel version of the activation maps.

Generate the final 2 x 2 activation-map figure from the archived maps:

```bash
python plot_acttime.py \
  --indir out_corrected \
  --cases all \
  --cmap turbo \
  --out figure_activation_maps_2x2.pdf \
  --dpi 300
```

Use `.png` instead of `.pdf` in `--out` to create the raster version. This
command was checked against the submitted PNG and reproduces it pixel for
pixel in the original environment.

## Protocol encoded in the driver

- Slab: 20 mm x 10 mm.
- Circular cavity radii: 1.2, 1.5, and 1.0 mm.
- Uniform fibre direction: 45 degrees.
- Field direction: +x, applied only at the vessel walls.
- Pulse duration: 3 ms; simulation duration: 30 ms.
- Mesh spacing: 0.2 mm; time step: 0.01 ms.
- Field sweep: 0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, and 5.0 V/cm.
- Activation threshold: -20 mV.

Conductivities in the source code are expressed in mS/um. The four cases are
isotropic monodomain, anisotropic monodomain, equal-anisotropy-ratio bidomain,
and unequal-anisotropy-ratio bidomain.

## Archived outputs

The `out_corrected/` directory contains the exact CSV and map arrays used for
the supplied figures. The `figures/` directory contains the final PDF and PNG
versions of both figures.

