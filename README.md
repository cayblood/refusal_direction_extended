# refusal-direction-extended

This project seeks to reproduce the findings of Arditi et al.'s "[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)". It also extends those findings by comparing refusal directions across Qwen2.5 Instruct model sizes and testing whether an alignment-based transfer works between them.

## Models

The primary experiment uses:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Use the Instruct variants for the primary experiments. Refusal behavior is a chat/instruction-tuning behavior, so Instruct models are the right default for reproducing harmful-vs-benign refusal activations. Qwen is public on Hugging Face, so it is also a good substitute while waiting for Meta Llama access approval.

## Runtime expectation

GPU-dependent tasks assume an active Colab Pro GPU runtime unless the task name or command explicitly says local. In practice:

- `mise baseline` expects CUDA and is intended to run in Colab Pro.
- `mise gpu-check` verifies the active runtime has CUDA.
- `mise baseline-local` is the explicit opt-in for local CPU/MPS/CUDA testing.

Model downloads are cached in the active runtime's Hugging Face cache. In Colab, that usually means the runtime cache unless you configure Hugging Face or your notebook to use mounted Drive storage.

Use `notebooks/baseline_colab.ipynb` as the Colab Pro entry point for GPU setup, dependency installation, model downloads, and baseline generation.

## Prerequisites

1. Install [`mise`](https://mise.jdx.dev/) in the environment where you run tasks.
2. For GPU-dependent tasks, use a Colab Pro runtime with GPU enabled.
3. Optional but recommended: log in to Hugging Face to avoid anonymous download rate limits.

Use `mise <task>` for project tasks. Use `mise exec -- ...` only when running tools directly, such as `uv` or `python`.

## Common tasks

```sh
mise tasks              # list available tasks
mise setup              # sync Python dependencies into the active environment
mise check-env          # print Python and core package versions
mise gpu-check          # verify CUDA is available in the active runtime
mise lint               # lint Python files with Ruff
mise lint-fix           # apply safe Ruff lint fixes
mise format             # format Python files with Ruff
mise format-check       # check formatting without modifying files
mise check              # run non-download, non-GPU checks: lint + format-check
mise hf-login           # optional: log in to Hugging Face to avoid rate limits
mise hf-whoami          # show the active Hugging Face account/token status
mise download-models    # download/cache both Qwen2.5 Instruct models
mise download-qwen-1_5b # download/cache only Qwen2.5 1.5B Instruct
mise download-qwen-3b   # download/cache only Qwen2.5 3B Instruct
mise baseline           # run baseline generations on CUDA / Colab Pro
mise baseline-local     # explicitly run baseline locally on CPU/MPS/CUDA
```

If `download-models` or `baseline` reports a network or rate-limit error, try `mise hf-login` and rerun the task.

## Editor linting

Ruff is the project linter and formatter. Zed workspace settings in `.zed/settings.json` enable the Ruff language server for Python linting and use Ruff as the Python formatter. Pyright remains configured for basic type/import analysis through `pyrightconfig.json`, with noisy lint-like type warnings disabled so they do not duplicate Ruff.

## Colab notebook

Open `notebooks/baseline_colab.ipynb` in Colab, enable a GPU runtime, and run the cells top to bottom. The notebook uses `uv` directly because Colab runtimes are ephemeral and do not need the local `mise` shell integration.

## Direct baseline command

The `baseline` task runs:

```sh
mise exec -- uv run python scripts/baseline.py --device cuda
```

By default the script runs both Qwen2.5 Instruct models and requires CUDA. Useful options:

```sh
mise exec -- uv run python scripts/baseline.py --max-new-tokens 256
mise exec -- uv run python scripts/baseline.py --model Qwen/Qwen2.5-1.5B-Instruct
mise exec -- uv run python scripts/baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --model Qwen/Qwen2.5-3B-Instruct
mise exec -- uv run python scripts/baseline.py --allow-local --device cpu --dtype float32
```
