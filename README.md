# refusal-direction-extended

This project seeks to reproduce the findings of Arditi et al.'s "[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)". It also extends those findings by comparing refusal directions across Llama 3.2 Instruct model sizes and testing whether an alignment-based transfer works between them.

## Models

The primary experiment uses:

- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

We use the Instruct variants for the primary experiments. Refusal behavior is a chat/instruction-tuning behavior, so Instruct models are the right default for reproducing harmful-vs-benign refusal activations.

## Pipeline outputs

The reproduction runs in stages, each producing artifacts the next stage reads:

1. `prepare-datasets` → `data/refusal_datasets/` (length-matched harmful/benign
   prompts formatted with the model chat template).
2. `collect-activations` → `data/activations/<model>/resid_post.pt`
   (residual-stream activations at the last several post-instruction token
   positions for every layer, shape
   `[n_prompts, n_positions, n_layers, d_model]`) plus
   `artifacts/activations/<model>/divergence_summary.json` and `divergence.png`
   (per-layer cosine similarity between mean harmful and mean benign activations
   at the final token — a sanity check that the two classes separate in the
   deeper layers, which is where refusal is expected to be mediated).
3. `extract-directions` → `data/activations/<model>/directions.pt` (a
   normalized difference-in-means candidate refusal direction for every
   `(position, layer)` pair, computed on a train split) plus
   `artifacts/activations/<model>/directions_summary.json`.
4. `evaluate-ablation` → `artifacts/activations/<model>/ablation_*.json`
   (refusal bypass rates from projecting each candidate direction out of the
   residual stream, ranked to pick the most causally effective layer/position,
   with example completions).

## Runtime expectation

GPU-dependent tasks assume CUDA unless the task name or command explicitly says local. You can run CUDA tasks in Colab Pro, Runpod, or another GPU runtime. In practice:

- `mise baseline` expects CUDA in the active shell environment.
- `mise gpu-check` verifies the active environment has CUDA.
- `mise baseline-local` is the explicit opt-in for local CPU/MPS/CUDA testing.
- `mise runpod-*` tasks execute on a configured Runpod pod over SSH.

Model downloads are cached in the active runtime's Hugging Face cache. In Colab, that usually means the runtime cache unless you configure Hugging Face or your notebook to use mounted Drive storage. In Runpod, use persistent storage if you want model caches to survive pod replacement.

Use `notebooks/baseline_colab.ipynb` as the Colab Pro entry point for GPU setup, dependency installation, model downloads, and baseline generation.

## Prerequisites

1. Install [`mise`](https://mise.jdx.dev/) in the environment where you run tasks.
2. For GPU-dependent tasks, use a Colab Pro runtime with GPU enabled.
3. Log in to Hugging Face with an account that has access to the gated Llama repositories.

Use `mise <task>` for project tasks. Use `mise exec -- ...` only when running tools directly, such as `uv` or `python`.

## Common tasks

Tasks that run model code default to CUDA in the active runtime. Tasks with a
`-local` suffix are the explicit opt-in for running on CPU/MPS/CUDA locally.

```sh
# Environment and code quality
mise tasks              # list available tasks
mise setup              # sync Python dependencies into the active environment
mise check-env          # print Python and core package versions
mise gpu-check          # verify CUDA is available in the active runtime
mise lint               # lint Python files with Ruff
mise lint-fix           # apply safe Ruff lint fixes
mise format             # format Python files with Ruff
mise format-check       # check formatting without modifying files
mise check              # run non-download, non-GPU checks: lint + format-check

# Hugging Face and model downloads
mise hf-login           # log in to Hugging Face for gated Llama access
mise hf-whoami          # show the active Hugging Face account/token status
mise download-models    # download/cache both Llama 3.2 Instruct models
mise download-model     # alias for download-models
mise download-llama-1b  # download/cache only Llama 3.2 1B Instruct
mise download-llama-3b  # download/cache only Llama 3.2 3B Instruct

# Pipeline (active runtime; *-local opts into a local CPU/MPS/CUDA run)
mise prepare-datasets   # prepare AdvBench/Alpaca datasets locally
mise smoke-datasets     # short generation smoke test on prepared datasets
mise baseline           # run baseline generations on CUDA in active shell
mise baseline-local     # explicitly run baseline locally on CPU/MPS/CUDA
mise collect-activations # collect last-token resid_post activations on CUDA
mise collect-activations-local # collect activations locally (CPU/MPS/CUDA)
mise smoke-activations  # collect activations for a few prompts as a smoke test
mise extract-directions # difference-in-means candidate refusal directions
mise evaluate-ablation  # validate directions by ablation on CUDA
mise evaluate-ablation-local # validate directions by ablation locally
mise evaluate-quantitative # ablation+addition 2x2 on the test split (CUDA)
mise evaluate-quantitative-local # ablation+addition 2x2 locally
mise plot-quantitative  # plot the addition sweep and 2x2 table (CPU)
mise evaluate-transfer  # cross-scale linear transfer of the direction (CUDA)
mise evaluate-transfer-local # cross-scale linear transfer locally
mise prepare-generic    # prepare the independent generic instruction set (control)
mise collect-generic-activations # collect generic-prompt activations (CUDA)
mise collect-generic-activations-local # collect generic activations locally
mise evaluate-transfer-independent # transfer control on the independent set (CUDA)
mise evaluate-transfer-independent-local # transfer control locally

# Runpod (execute the corresponding step on the configured pod over SSH)
mise runpod-check-config # print configured Runpod connection settings
mise runpod-check-ephemeral-config # print API-based pod spec
mise runpod-check-h100-config # print persistent H100 pod spec
mise runpod-persistent-h100 # create/reuse network volume and leave H100 pod running
mise runpod-sync        # rsync this repository to the Runpod pod
mise runpod-setup       # install uv and sync dependencies on Runpod
mise runpod-gpu-check   # verify CUDA on Runpod
mise runpod-download-models # download/cache models on Runpod
mise runpod-prepare-datasets # prepare AdvBench/Alpaca datasets on Runpod
mise runpod-smoke-datasets # short generation smoke test on prepared datasets on Runpod
mise runpod-baseline    # run baseline on Runpod
mise runpod-collect-activations # collect activations for both models on Runpod
mise runpod-smoke-activations # collect activations for a few prompts on Runpod
mise runpod-extract-directions # extract candidate refusal directions on Runpod
mise runpod-evaluate-ablation # validate directions by ablation on Runpod
mise runpod-evaluate-quantitative # ablation+addition 2x2 on Runpod
mise runpod-evaluate-transfer # cross-scale linear transfer on Runpod
mise runpod-prepare-generic # prepare the independent generic set on Runpod
mise runpod-collect-generic-activations # collect generic-prompt activations on Runpod
mise runpod-evaluate-transfer-independent # transfer control on Runpod
mise runpod-pull-artifacts # pull generated plots/summaries from Runpod to local
mise runpod-all         # sync, setup, download models, and run baseline
mise runpod-ephemeral   # create pod, run baseline, terminate pod
mise runpod-terminate   # terminate the configured pod (network volume preserved)
```

If `download-models`, `prepare-datasets`, or `baseline` reports an authorization, network, or rate-limit error, run `mise hf-login` with an account that has Llama access and rerun the task.

## Editor linting

Ruff is the project linter and formatter. Zed workspace settings in `.zed/settings.json` enable the Ruff language server for Python linting and use Ruff as the Python formatter. Pyright remains configured for basic type/import analysis through `pyrightconfig.json`, with noisy lint-like type warnings disabled so they do not duplicate Ruff.

## Colab notebook

Open `notebooks/baseline_colab.ipynb` in Colab, enable a GPU runtime, and run the cells top to bottom. The notebook uses `uv` directly because Colab runtimes are ephemeral and do not need the local `mise` shell integration.

## Runpod workflow

### Persistent H100 pod + network volume

This is the preferred Runpod workflow. It creates or reuses a persistent network volume, retries H100 pod creation until a usable SSH/CUDA pod is running, and leaves the successful pod running for repeated experiments.

Configure your API key, Hugging Face token, and preferred Runpod datacenter:

```sh
export RUNPOD_API_KEY=...
export HF_TOKEN=hf_...
export RUNPOD_DATACENTER_ID=US-CA-2
```

Then create or reconnect to the persistent H100 pod:

```sh
mise runpod-persistent-h100
```

The task writes `.runpod.env` in the project root with the pod ID, SSH host/port, datacenter, and network volume ID. This file is ignored by git and is read automatically by later `mise runpod-*` tasks, so you do not need to repeatedly export pod connection settings.

Useful optional settings:

```sh
export RUNPOD_NETWORK_VOLUME_ID=...      # reuse an existing volume
export RUNPOD_NETWORK_VOLUME_SIZE_GB=200 # default: 200
export RUNPOD_RETRY_SLEEP_SECONDS=60     # default: 60
export RUNPOD_MAX_ATTEMPTS=0             # default: 0 means retry forever
```

After the pod is ready:

```sh
mise runpod-sync
mise runpod-setup
mise runpod-download-models
mise runpod-prepare-datasets
mise runpod-baseline
mise runpod-collect-activations
mise runpod-extract-directions
mise runpod-evaluate-ablation
mise runpod-evaluate-quantitative
mise runpod-evaluate-transfer
mise runpod-prepare-generic
mise runpod-collect-generic-activations
mise runpod-evaluate-transfer-independent
mise runpod-pull-artifacts
mise runpod-terminate
```

`runpod-collect-activations` writes the large activation tensors to
`data/activations/<model>/` on the pod (kept there for the direction-extraction
step) and small summary/plot artifacts to `artifacts/activations/<model>/`. The
later stages (`extract-directions` through `evaluate-transfer-independent`) read
those cached tensors, so they reuse the collection rather than recomputing it.
`runpod-pull-artifacts` rsyncs the `artifacts/` tree back to the local repo so
the plots and summaries are available for the writeup. The activation caches are
intentionally excluded from `runpod-sync` so a later sync never deletes them from
the pod.

Failed pod creation attempts are terminated automatically. A successfully verified H100 pod is intentionally left running so you can rerun tests frequently; run `mise runpod-terminate` when you are done to stop GPU billing (the network volume, and the model/activation caches on it, are preserved).

### Existing pod

```sh
export RUNPOD_SSH_TARGET=root@203.0.113.10
export RUNPOD_SSH_PORT=22
export HF_TOKEN=hf_...
mise runpod-all
```

### Ephemeral pod

```sh
export RUNPOD_API_KEY=...
export HF_TOKEN=hf_...
mise runpod-ephemeral
```
