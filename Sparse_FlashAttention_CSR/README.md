# CSR Attention Mask

Users may access our CSR mask attention implementation by first importing the library in the following manner:

```python
import torch
import spfa_csr
```

Please note that the PyTorch import is required to use this library. The function is compatible with both `torch.float32` and `torch.float16` data types and can be called by

```python
spfa_csr.forward(q, k, v, w_row_off, w_col_ind, w_val, m, l, out, use_nan)
```

where the arguments are as followed:

- **`q`** is the query matrix.

- **`k`** is the key matrix.

- **`v`** is the value matrix.

- **`w_row_off`** is the row offset vector of the mask's CSR form.

- **`w_col_ind`** is the column index vector of the mask's CSR form.

- **`w_val`** is the value vector of the mask's CSR form.

- **`m`** is a 0-initialized vector of length context length that stores the maximum values for the online softmax.

- **`l`** is a `-inf`-initialized vector of length context length that stores the normalization terms for the online softmax.

- **`out`** is the output matrix.

- **`use_nan`** is a boolean flag that indicates whether fully masked rows will be set to `NaN` or not. If set to `true`, then the values will be `NaN`. If set to `false`, then the values will be `0.0`.