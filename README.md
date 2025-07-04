# cslrtools

## Overview
cslrtools is a Python package that provides a set of tools related to Continuous Sign Language Recognition (CSLR). It mainly includes features such as attention mask generation, convolution output size calculation, CTC decoding, and word error rate (WER) computation.

## Directory Structure
```
pyproject.toml
README.md
uv.lock
src/
  cslrtools/
    __init__.py
    attn_mask.py
    convsize.py
    ctc_decode.py
    py.typed
    utils.py
    word_error_rate.py
```

## Installation
1. Clone the repository.
2. Install the required packages.

```bash
pip install .
```

Alternatively, use `pyproject.toml` or `uv.lock` to install dependencies.

## Usage
Each module can be imported from the `cslrtools` package.

Example:
```python
import torch.nn as nn
from cslrtools import attn_mask, convsize, ctc_decode, word_error_rate

# Generate attention mask
mask = attn_mask.attn_mask(...)

# Compute convolution output size
conv2d = nn.Conv2d(...)
size = convsize.ConvSize(conv2d)

# CTC decoding
decoded = ctc_decode.ctc_decode(...)

# Calculate word error rate
wer = word_error_rate.wer(...)
```

For detailed usage of each function, please refer to the docstrings in the source code.

## License
This project is licensed under the Apache License 2.0.

See the LICENSE file for details.
