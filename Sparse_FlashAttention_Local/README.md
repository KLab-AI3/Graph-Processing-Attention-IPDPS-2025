# Local Attention Mask

Users may access our local mask attention implementation by first importing the library in the following manner:

```python
import torch
import spfa_local
```

Please note that the PyTorch import is required to use this library. The function is compatible with both `torch.float32` and `torch.float16` data types and can be called by

```python
spfa_local.forward(q, k, v, m, l, out, LOCAL_SIZE)
```

where the arguments are as followed:

- **`q`** is the query matrix.

- **`k`** is the key matrix.

- **`v`** is the value matrix.

- **`m`** is a 0-initialized vector of length context length that stores the maximum values for the online softmax.

- **`l`** is a `-inf`-initialized vector of length context length that stores the normalization terms for the online softmax.

- **`out`** is the output matrix.

- **`LOCAL_SIZE`** is an integer parameter used to control the local window size. The total number of parameters that can be seen for a given token is calculated as follows: `1 + (LOCAL_SIZE * 2)`. This is essentially saying that the window parameter is the number of tokens a given token can look ahead and backwards.