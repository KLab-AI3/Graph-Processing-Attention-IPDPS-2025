#include <vector>

#include <torch/extension.h>


// Verification macros.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// C++ dispatch function declaration.
torch::Tensor spfa_bsr_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor W_block_row_off,
    torch::Tensor W_block_col_ind,
    torch::Tensor W_val,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor O, 
    int block_length,
    bool use_nan
);


// C++ interface function for PyTorch inputs.
torch::Tensor spfa_bsr_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor W_block_row_off,
    torch::Tensor W_block_col_ind,
    torch::Tensor W_val,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor O,
    int block_length,
    bool use_nan
) {
  // Verify each of the inputs.
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  CHECK_INPUT(W_block_row_off);
  CHECK_INPUT(W_block_col_ind);
  CHECK_INPUT(W_val);
  CHECK_INPUT(m);
  CHECK_INPUT(l);
  CHECK_INPUT(O);

  // Return the result of the dispatch function.
  return spfa_bsr_cuda_forward(Q, K, V, W_block_row_off, W_block_col_ind, W_val, m, l, O, block_length, use_nan);
}


// Binding function for Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spfa_bsr_forward, "Sparse-FlashAttention forward (CUDA)");
}