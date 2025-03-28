# COO Attention Mask

Users may access our COO mask attention implementation by first importing the library in the following manner:

```python
import torch
import spfa_coo
```

Please note that the PyTorch import is required to use this library. The function is compatible with both `torch.float32` and `torch.float16` data types and can be called by

```python
spfa_coo.forward(q, k, v, w_row_ind, w_col_ind, w_val, m, l, out, use_nan)
```

where the arguments are as followed:

- **`q`** is the query matrix.

- **`k`** is the key matrix.

- **`v`** is the value matrix.

- **`w_row_ind`** is the row index vector of the mask's COO form.

- **`w_col_ind`** is the column index vector of the mask's COO form.

- **`w_value`** is the value vector of the mask's COO form.

- **`m`** is a `-inf`-initialized vector of length context length that stores the maximum values for the online softmax.

- **`l`** is a `0.0`-initialized vector of length context length that stores the normalization terms for the online softmax.

- **`out`** is the output matrix.

- **`use_nan`** is a boolean flag that indicates whether fully masked rows will be set to `NaN` or not. If set to `true`, then the values will be `NaN`. If set to `false`, then the values will be `0.0`.