# Local 2D Dilated Attention Mask

Users may access our local 2D dilated mask attention implementation by first importing the library in the following manner:

```python
import torch
import spfa_local_2d_dilated
```

Please note that the PyTorch import is required to use this library. The function is compatible with both `torch.float32` and `torch.float16` data types and can be called by

```python
spfa_local_2d_dilated.forward(q, k, v, m, l, out, BLOCK_SIZE, DILATE_FACTOR, use_nan)
```

where the arguments are as followed:

- **`q`** is the query matrix.

- **`k`** is the key matrix.

- **`v`** is the value matrix.

- **`m`** is a `-inf`-initialized vector of length context length that stores the maximum values for the online softmax.

- **`l`** is a `0.0`-initialized vector of length context length that stores the normalization terms for the online softmax.

- **`out`** is the output matrix.

- **`BLOCK_SIZE`** is an integer parameter used to control the local block size. Block sizes must be integer divisors of the context length. Blocks fall along the upper left/lower right diagonal.

- **`DILATE_FACTOR`** is an integer value within [0, `BLOCK_SIZE`] that specifies the amount of horizontal/vertical space between each token of a block.

- **`use_nan`** is a boolean flag that indicates whether fully masked rows will be set to `NaN` or not. If set to `true`, then the values will be `NaN`. If set to `false`, then the values will be `0.0`.