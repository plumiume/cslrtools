# Copilot Instructions for cslrtools

## Project Overview
- **Purpose:** Python toolkit for Continuous Sign Language Recognition (CSLR) research and development.
- **Core Modules:**
  - `attn_mask.py`: Attention mask generation for sequence models.
  - `convsize.py`: Convolution and transposed convolution output size calculations.
  - `ctc_decode.py`: CTC decoding utilities for sequence-to-sequence tasks.
  - `word_error_rate.py`: Edit distance and word error rate (WER) computation.
  - `utils.py`: Shared utilities for batch shape checking and product calculations.
  - `dataset/`: Data handling for PyTorch and Lightning, including advanced v2 abstractions.

## Architecture & Patterns
- **Package Structure:** All main logic is under `src/cslrtools/`. Import modules via `from cslrtools import ...`.
- **Type Safety:** Uses type hints and `py.typed` for strict type checking. Type stubs for `safetensors` in `typings/`.
- **Data Abstractions:**
  - `dataset/pytorch.py`: Defines `Dataset` and `DataTuple` for standardized batch handling.
  - `dataset/lightning.py`: Extends PyTorch datasets for Lightning integration.
  - `dataset/v2/torch/`: Advanced dataset and loader abstractions, with internal call guards and caching.
- **Functional Patterns:** Many utilities are pure functions; batch operations expect `Tensor` inputs and lengths.
- **Error Handling:** Batch size mismatches and missing dependencies (e.g., Lightning) raise explicit errors.

## Developer Workflows
- **Installation:**
  - Standard: `pip install .`
  - With Lightning: `pip install .[lightning]`
  - Dependencies managed via `pyproject.toml` and `uv.lock`.
- **Testing:**
  - No explicit test runner found; recommend using `pytest` for new tests.
  - Validate with type checkers (e.g., `pyright`, `mypy`).
- **Debugging:**
  - Use docstrings and type hints for function usage.
  - For dataset issues, check batch shapes and metadata types.

## Conventions & Integration
- **Licensing:** Apache 2.0. All source files include license headers.
- **External Dependencies:**
  - Core: `torch`, `torchaudio`, `torchvision`, `safetensors`, `zarr`.
  - Optional: `lightning` for advanced training workflows.
- **Import Patterns:**
  - Prefer relative imports within `src/cslrtools/`.
  - Expose public API via `__init__.py` using `__all__`.
- **Type Stubs:**
  - Custom stubs for `safetensors` in `typings/safetensors/`.

## Examples
```python
from cslrtools import attn_mask, convsize, ctc_decode, word_error_rate
mask = attn_mask.attn_mask(lengths)
size = convsize.ConvSize(conv2d)
decoded = ctc_decode.ctc_decode(x, xlen)
wer = word_error_rate.wer(a, alen, b, blen)
```

## Key Files & Directories
- `src/cslrtools/`: Main package modules
- `src/cslrtools/dataset/`: Data abstractions
- `typings/safetensors/`: Type stubs
- `pyproject.toml`: Dependency management
- `README.md`: Usage and architecture summary

---
**Feedback:** Please review and suggest additions or corrections for unclear or missing sections.
