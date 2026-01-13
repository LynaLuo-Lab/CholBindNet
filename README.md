### CholBindNet Docker Usage

This repository provides a Docker image that runs `EvaluateFile.py` to evaluate cholesterol-binding models (GAT, GCN, and GNN) against input PDB files.

The container image is published to GitHub Container Registry (GHCR).

#### Image name

```
ghcr.io/lynaluo-lab/cholbindnet:main
```

#### Pull the image

```
docker pull ghcr.io/lynaluo-lab/cholbindnet:main
```

#### Run with default settings (uses sample data inside the image)

The container’s entrypoint is set to:

```
python EvaluateFile.py
```

So a simple `docker run` will execute the evaluator using the defaults defined in `EvaluateFile.py`:

```
docker run --rm -it ghcr.io/lynaluo-lab/cholbindnet:main
```

Defaults in `EvaluateFile.py`:
- `--test_files_path` = `TestEvaluationFiles`
- `--output_dir` = `CholBindOutput`
- `--gat_models_path` = `GAT_Model`
- `--gnn_models_path` = `GNN_Model`
- `--gcn_models_path` = `GCN_Model`

These directories are included within the image, so the above command will run end-to-end inside the container and produce outputs under `/app/CholBindOutput` in the container.

#### Run on your own data (bind-mount host directories)

Mount a host directory with your `.pdb` files and a host output directory, then override the CLI arguments to point inside the container:

```
docker run --rm -it \
  -v /path/to/your/pdbs:/data \
  -v /path/to/output:/out \
  ghcr.io/lynaluo-lab/cholbindnet:main \
  --test_files_path /data \
  --output_dir /out
```

If you keep the model folders on the host instead of using the built-in ones, mount them as well and override their paths:

```
docker run --rm -it \
  -v /path/to/your/pdbs:/data \
  -v /path/to/output:/out \
  -v /path/to/GAT_Model:/models/gat \
  -v /path/to/GNN_Model:/models/gnn \
  -v /path/to/GCN_Model:/models/gcn \
  ghcr.io/lynaluo-lab/cholbindnet:main \
  --test_files_path /data \
  --output_dir /out \
  --gat_models_path /models/gat \
  --gnn_models_path /models/gnn \
  --gcn_models_path /models/gcn
```

#### Build locally (optional)

If you want to build the image locally from this repository:

```
docker build -t cholbindnet:local .
docker run --rm -it cholbindnet:local
```

#### Notes

- The image is CPU-based (`python:3.10` base). No CUDA is included.

## Computational Resources and Cost

### Hardware Environment

All model training and evaluation experiments were conducted on a high-performance computing (HPC) cluster node with the following specifications:

**CPU**
- Processor: AMD EPYC 7763 (64 cores, 128 threads)
- Architecture: x86_64
- Base/Boost Frequency: 1.5–3.5 GHz
- Total CPU cores available per node: 128
- CPU cores used per experiment: up to **32**
- Cache hierarchy:
  - L1 Cache: 2 MB
  - L2 Cache: 32 MB
  - L3 Cache: 256 MB

**GPU**
- GPU Model: NVIDIA RTX A4500
- GPU Memory: 20 GB GDDR6
- CUDA Version: 12.4
- NVIDIA Driver Version: 550.90.07
- GPUs available per node: 4
- GPUs used per experiment: **1**

Training jobs were executed using a single GPU in combination with multi-threaded CPU preprocessing and data loading.

---

### Computational Cost and Runtime

Each training experiment consists of an ensemble of **50 submodels**, trained independently and aggregated to produce the final predictive model.

- GPU usage: 1 × NVIDIA RTX A4500
- CPU usage: up to 32 cores
- Typical wall-clock training time:
  - **~24–48 hours per experiment** (50 submodels)

Parallelization was applied at the level of CPU-based preprocessing and data loading, while GPU-accelerated model training was performed sequentially within a single job submission. No multi-GPU training was employed.

---

