# LANTERN_BASIC

Minimal setup to train two baseline calorimeter models:

- **Energy model** (DDPM): `configs/d2_energy_model_DDPM.yaml`
- **Shape model** (Diffusion): `configs/d2_shape_model_diffusion.yaml`

## 1) Create the Conda environment

We ship a single environment file: `LANTERN.yml`.

```bash
# From the repo root
conda env create -f LANTERN.yml
conda activate lantern
```

> If you already created it once, update with:
>
> ```bash
> conda env update -f LANTERN.yml --prune
> conda activate lantern
> ```

## 2) Point the configs to your data

Open these two files and **edit the paths** to your datasets and geometry:

- `configs/d2_energy_model_DDPM.yaml`
- `configs/d2_shape_model_diffusion.yaml`

Update the keys that reference your files (training, test, and XML). Example:

```yaml
# Example — adjust to your actual field names
data:
  hdf5_file: /abs/path/to/train.hdf5
  eval_hdf5_file:  /abs/path/to/test.hdf5
  xml_filename:   /abs/path/to/geometry.xml
```

> Tip: search within each YAML for `train`, `test`, `path`, or `xml` and replace the placeholder paths.

## 3) Training order (important)

1. **Train the Energy model first** using `configs/d2_energy_model_DDPM.yaml`.
2. After the energy model finishes, open `configs/d2_shape_model_diffusion.yaml` and set the **folder path** to the trained energy model (the directory that contains its checkpoints). For example:
   
```yaml
# inside d2_shape_model_diffusion.yaml
energy_model:
  dir: /abs/path/to/energy_model   # folder with trained energy-model checkpoints
```
   
> The exact key name may be `energy_model`, `energy_model_dir`, or similar in your config—set it to the **folder** produced by the energy run.

## 4) Run on Slurm

From the repo root (after editing config paths and setting the energy model folder as above):

```bash
# Step 1 — train energy model
sbatch sbatch_energy_model.sh    # uses configs/d2_energy_model_DDPM.yaml

# Step 2 — train shape model (after energy model is trained)
sbatch sbatch_shape_model.sh     # uses configs/d2_shape_model_diffusion.yaml
```

> Open each `.sh` to confirm the config it calls and your account/partition settings.

## Repo layout (high level)

```
LANTERN_BASIC/
├─ configs/
│  ├─ d2_energy_model_DDPM.yaml              # EDIT PATHS
│  └─ d2_shape_model_diffusion.yaml          # EDIT PATHS + ENERGY MODEL FOLDER
├─ src/                                      # training code
├─ LANTERN.yml                               # conda environment
├─ sbatch_energy_model.sh                     # Slurm launcher
└─ sbatch_shape_model.sh                      # Slurm launcher
```

## Troubleshooting

- **Conda env fails to solve** → update conda/mamba, then `conda env update -f LANTERN.yml --prune`.
- **File not found** → double-check absolute paths in the two YAML files.
- **Geometry/XML errors** → ensure the `xml_path` (or equivalent key) points to a valid file accessible on your machine/cluster.
- **Slurm**: verify your account/partition settings inside the `sbatch_*.sh` scripts.
