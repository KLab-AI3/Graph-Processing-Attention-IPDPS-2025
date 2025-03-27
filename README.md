# Graph-Processing-Attention-IPDPS-2025

This repo contains code associated with the paper: "Longer Attention Span: Increasing Transformer Context Length with Sparse Graph Processing Techniques". It will be presented at IPDPS 2025 and the text can be found at: [https://www.arxiv.org/abs/2502.01659](https://www.arxiv.org/abs/2502.01659)


# Principle Idea

The self-attention mechanism within transformers suffers from quadratic complexity for both memory and compute. We take inspiration from FlashAttention's application of the online softmax to reduce the memory complexity to a linear one dependent on the context length. Additionally, with the creation of many mask types for the attention matrix, our explicit mask implementations are able to map any arbitrary mask to the GPU without excess calculation (i.e. utilizing tensor cores to calculate a non-dense block and masking the result). The reduction in memory complexity and, in sparse scenarious, compute complexity is designed to allow for users to achieve longer context length on the same hardware while also performing decreasing the runtime of their masked attention.

We define two mask types: explicit and implicit. An explicit mask is one that resides in memory; we give the option for storing a mask in CSR and COO form. Implcit masks are those that are defined by parameters, meaning the bounds/indices for calculation are calculated within the kernel itself. The libraries are written in CUDA, however there are bindings and they are designed to be used within PyTorch. Please note that this implementation is currently limited to a single GPU with single-headed, single-batched attention.


# Pre-Requisites

- **NVIDIA GPU:** The current implementation can only be run on NVIDIA hardware.

- **Python:** We utilized Python version 3.10.8 and recommend either that or a newer version.

- **PyTorch:** The libraries are dependent on PyTorch, we recommend using a version >= 2.4.0 that is compiled with a compatible CUDA version to that installed on your system.

- **CUDA:** The version should be compatible with your PyTorch install above.

- **Ninja:** This will be used in the install process. We utilized Ninja version 1.11.1 and recommend either that or a newer version.


# Installation

1. Copy the 6 folders into your `../python/site-packages/torch/` directoy.

2. Determine the compute capability of your NVIDIA GPU (i.e. A100 = 8.0).

3. Within each of the 6 directories, run the following command to compile/install the library (inserting your compute capability in place of **\$CC\$**): `TORCH_CUDA_ARCH_LIST="$CC$" python setup.py install`.

4. Verify the libraries are installed properly and functioning by running the verification script in the `/verify/` directory: `python verify.py`.


# Attention Mask Implementations

**The function names and parameters are specified within the associated folders.**

- **COO:** located within `/Sparse_FlashAttention_COO/`, this will allow the user to perform the attention operation using any arbitrary mask that is stored in the COO format.

- **CSR:** located within `/Sparse_FlashAttention_CSR/`, this will allow the user to perform the attention operation using any arbitrary mask that is stored in the CSR format.

- **Global (no local):** located within `/Sparse_FlashAttention_Global_No_Local/`, this will allow the user to specify a list of global attention tokens that will not be masked, except for a local window region. Everything outside of the provided row/column indices is masked.

- **Local:** located within `/Sparse_FlashAttention_Local/`, this will allow the user to specify a 1D width parameter to control how wide a local context window should be.

- **Local 1D:** located within `/Sparse_FlashAttention_Local_1D_Dilated/`, this will allow the user to specify a 1D width parameter and dilation parameter to control the width and dilation spacing of a local context window.

- **Local 2D:** located within `/Sparse_FlashAttention_Local_2D_Dilated/`, this will allow the user to specify a 2D block size parameter and dilation parameter to control the block size and dilation spacing of a block-diagonal mask (2D dilation).