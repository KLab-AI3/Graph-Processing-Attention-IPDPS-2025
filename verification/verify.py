import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import spfa_bsr
import spfa_csr
import spfa_coo
import spfa_global_no_local
import spfa_local
import spfa_local_1d_dilated
import spfa_local_2d_dilated

import math
import random
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# Store the specified dimensions.
Q = 256
K = 256
V = 256
D = 32

# Specify the level density of the mask (1 -> fully dense, 0 -> fully sparse).
DENSE_LEVEL = 1

# Set the default device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Control the data type of individual tensors.
data_type = torch.float32

# Verify that the default data type for tensor creation is float32 (single-precision floating-point).
torch.set_default_dtype(torch.float32)

# Create random tensors of the specified dimensions.
q = torch.rand([Q, D], device = device, dtype = data_type)
k = torch.rand([K, D], device = device, dtype = data_type)
v = torch.rand([V, D], device = device, dtype = data_type)

# Create random connection of the provided sparsity level for the mask.
mask = torch.rand((Q, Q), device = device, dtype = data_type) < DENSE_LEVEL

# Clone the mask.
mask_2 = torch.clone(mask)
mask_3 = torch.clone(mask)
mask_4 = torch.clone(mask)
mask_5 = torch.clone(mask)

# Convert the PyTorch mask to float to use 
# mask = mask.type(torch.float32)
# mask = torch.where(mask == 1.0, 0.0, -float("inf"))

# Calculate the PyTorch attention result.
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
with sdpa_kernel(SDPBackend.MATH):
# with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    torch.cuda.synchronize()

# NOTE: Restrict the mask to int32
mask = mask.type(data_type)
csr_mask = mask.to_sparse_csr()
w_row_off = csr_mask.crow_indices().type(torch.uint64)
w_col_ind = csr_mask.col_indices().type(torch.uint32)
w_val = csr_mask.values()

o_csr = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

o_csr = spfa_csr.forward(q, k, v, w_row_off, w_col_ind, w_val, m, l, o_csr, True)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical
if torch.allclose(torch_out, o_csr, equal_nan = True):
    print("The SPFA-CSR results match PyTorch's.")
else:
    print("FAIL")

print("Max difference from PyTorch (SPFA-CSR):", torch.max(o_csr - torch_out))
print("Min difference from PyTorch (SPFA-CSR):", torch.min(o_csr - torch_out))


# NOTE: Restrict the mask to int32
mask_2 = mask_2.type(data_type)
coo_mask = mask_2.to_sparse_coo()
w_row_ind = coo_mask.indices()[0].type(torch.uint64)
w_col_ind = coo_mask.indices()[1].type(torch.uint32)
w_val = coo_mask.values()

o_coo = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

o_coo = spfa_coo.forward(q, k, v, w_row_ind, w_col_ind, w_val, m, l, o_coo, True)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out, o_coo, equal_nan = True):
    print("The SPFA-COO results match PyTorch's (dense).")
else:
    print("FAIL")

print("Max difference from dense PyTorch (SPFA-COO):", torch.max(o_coo - torch_out))
print("Min difference from dense PyTorch (SPFA-COO):", torch.min(o_coo - torch_out))

# Verify that the CSR and COO outputs are identical.
if torch.allclose(o_csr, o_coo):
    print("The SPFA-CSR results match SPFA-COO's.")
else:
    print("FAIL")


# Set the block size (note that the heigh/first dimension is always 1, these are slices 
# as we parallelize along the sequence length axis).
BLOCK_SIZE = 32

# Setup the mask for the BSR format.
mask_3 = mask_3.type(data_type)
bsr_mask = mask_3.to_sparse_bsr(blocksize=(1, BLOCK_SIZE))
w_block_row_off = bsr_mask.crow_indices().type(torch.uint64)
w_block_col_ind = bsr_mask.col_indices().type(torch.uint32)
w_val = bsr_mask.values()

# New inputs (and output) for the BSR operation.
o_bsr = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Perform the SPFA-BSR operation.
o_bsr = spfa_bsr.forward(q, k, v, w_block_row_off, w_block_col_ind, w_val, m, l, o_bsr, BLOCK_SIZE, True)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out, o_bsr, equal_nan = True):
    print("The SPFA-Local results match PyTorch's.")
else:
    print("FAIL")

print("Max difference from PyTorch (SPFA-BSR):", torch.max(o_bsr - torch_out))
print("Min difference from PyTorch (SPFA-BSR):", torch.min(o_bsr - torch_out))

# Verify that the Local and CSR outputs are identical, meaning COO as well.
if torch.allclose(o_bsr, o_csr, equal_nan = True):
    print("The SPFA-BSR results match SPFA-CSR's (and COO's).")
else:
    print("FAIL")

# Setup for a non-dense operation where only regions of block size are masked or not. Please note that 
# masking within a black that has non-zeros (True's) as well will not work with this implementation.
for i in range(Q):
    # This row is fully masked out.
    mask_4[0][i] = False

for i in range(BLOCK_SIZE):
    # This row has some blocks masked and others not.
    mask[2][i] = False
    mask[2][i + (3 * BLOCK_SIZE)] = False

# Calculate the PyTorch attention result.
with sdpa_kernel(SDPBackend.MATH):
    torch_out_bsr = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_4)
    torch.cuda.synchronize()

# Setup the mask for the BSR format.
mask_4 = mask_4.type(data_type)
bsr_mask = mask_4.to_sparse_bsr(blocksize=(1, BLOCK_SIZE))
w_block_row_off = bsr_mask.crow_indices().type(torch.uint64)
w_block_col_ind = bsr_mask.col_indices().type(torch.uint32)
w_val = bsr_mask.values()

# New inputs (and output) for the BSR operation.
o_bsr_2 = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Perform the SPFA-BSR operation.
o_bsr_2 = spfa_bsr.forward(q, k, v, w_block_row_off, w_block_col_ind, w_val, m, l, o_bsr_2, BLOCK_SIZE, True)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out_bsr, o_bsr_2, equal_nan = True):
    print("The SPFA-Local results match PyTorch's with a block sparse mask.")
else:
    print("FAIL")


# Set the distance a token can look in either direction for local attention (fully dense).
LOCAL_SIZE = Q - 1

# Set the new input tensors.
o_local = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Run the SPFA-Local operation.
o_local = spfa_local.forward(q, k, v, m, l, o_local, LOCAL_SIZE)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out, o_local, equal_nan = True):
    print("The SPFA-Local results match PyTorch's.")
else:
    print("FAIL")

print("Max difference from PyTorch (SPFA-Local):", torch.max(o_local - torch_out))
print("Min difference from PyTorch (SPFA-Local):", torch.min(o_local - torch_out))

# Verify that the Local and CSR outputs are identical, meaning COO as well.
if torch.allclose(o_local, o_csr, equal_nan = True):
    print("The SPFA-Local results match SPFA-CSR's (and COO's and BSR's).")
else:
    print("FAIL")

# Set the local size so that it is the identity matrix.
IDENTITY = 0

# Create a new output tensor and other new inputs.
o_local_2 = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Run the SPFA-Local operation on new inputs.
o_local_2 = spfa_local.forward(q, k, v, m, l, o_local_2, IDENTITY)
torch.cuda.synchronize()

# Verify that the Local output and v are identical (it multiplies the identity matrix).
if torch.allclose(o_local_2, v, equal_nan = True):
    print("The SPFA-Local output #2 (total window size = 1, look ahead/back = 0) result matches v.")
else:
    print("FAIL")

print("Max difference from V (SPFA-Local):", torch.max(o_local_2 - v))
print("Min difference from V (SPFA-Local):", torch.min(o_local_2 - v))


# Create an empty mask to hold the global attention (and anti-local).
mask_g = torch.full((Q, Q), False, device = device, dtype = torch.bool)

# Specify the global tokens.
globs = [0, 1, 10, 40]

# Bring the token list to the GPU.
globs = torch.Tensor(globs).cuda()
globs = globs.type(torch.int32)

# Mark tokens as masked.
for i in globs:
    mask_g[i, :] = True
    mask_g[:, i] = True

# Create a window of size 10.
WINDOW = 10

# Adjust for the non-local attention.
for i in range(Q):
    left_bound = max(0, i - WINDOW)
    right_bound = min(Q, i + WINDOW) + 1
    mask_g[i][left_bound:right_bound] = False

# Calculate the PyTorch attention result.
with sdpa_kernel(SDPBackend.MATH):
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_g)
    torch.cuda.synchronize()

# Setup the buffers.
o_glob = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Launch the global (non-local) kernel.
o = spfa_global_no_local.forward(q, k, v, m, l, o_glob, globs, WINDOW)
torch.cuda.synchronize()

# Verify that we match the PyTorch output for glboal (non-local) attention.
if torch.allclose(torch_out, o_glob):
    print("Global (non-local) attention matches PyTorch.")
else:
    print("FAIL")

# Print out the extrema differences.
print("Max difference from PyTorch (SPFA-Global (non-local)):", torch.max(o_glob - torch_out))
print("Min difference from PyTorch (SPFA-Global (non-local)):", torch.min(o_glob - torch_out))


# BigBird (CSR) Attention mask.
# Create a mask.
mask = torch.zeros((Q, Q), dtype=torch.bool, device=device)

# Create random connection of the provided sparsity level for the mask.
mask_2 = torch.rand((Q, Q), device = device, dtype = data_type) < 0.1

globs = random.sample(range(Q), 5)
globs.sort()

# Bring the token list to the GPU.
globs = torch.Tensor(globs).cuda()
globs = globs.type(torch.int32)

for i in globs:
    mask[i, :] = True
    mask[:, i] = True

for i in range(Q):
    start = max(0, i - WINDOW)
    end = min(Q, i + WINDOW + 1)
    mask[i, start:end] = True

# Create the total mask.
mask_3 = torch.where(mask == True, True, mask_2)

# Calculate the PyTorch attention result.
with sdpa_kernel(SDPBackend.MATH):
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_3)
    torch.cuda.synchronize()

# Create the buffers.
o_glob2 = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Create the CSR mask.
mask_g2 = mask_3.type(data_type)
csr_mask = mask_g2.to_sparse_csr()
w_row_off = csr_mask.crow_indices().type(torch.uint64)
w_col_ind = csr_mask.col_indices().type(torch.uint32)
w_val = csr_mask.values()

# Run the random attention using CSR form.
o_glob2 = spfa_csr.forward(q, k, v, w_row_off, w_col_ind, w_val, m, l, o_glob2, False)
torch.cuda.synchronize()

# Verify that we match the PyTorch output for BigBird attention.
if torch.allclose(torch_out, o_glob2, atol = 1e-1):
    print("BigBird (CSR) attention matches PyTorch.")
else:
    print("FAIL")

# Print out the extrema differences.
print("Max difference from PyTorch (BigBird (CSR) Attention):", torch.max(o_glob2 - torch_out))
print("Min difference from PyTorch (BigBird (CSR) Attention):", torch.min(o_glob2 - torch_out))


# BigBird (Multiple) Attention mask.
# Create a mask.
mask = torch.zeros((Q, Q), dtype=torch.bool, device=device)

# Create random connection of the provided sparsity level for the mask.
mask_2 = torch.rand((Q, Q), device = device, dtype = data_type) < 0.1

globs = random.sample(range(Q), 5)
globs.sort()

# Bring the token list to the GPU.
globs = torch.Tensor(globs).cuda()
globs = globs.type(torch.int32)

for i in globs:
    mask[i, :] = True
    mask[:, i] = True

for i in range(Q):
    start = max(0, i - WINDOW)
    end = min(Q, i + WINDOW + 1)
    mask[i, start:end] = True

# Create the total mask.
mask_3 = torch.where(mask == True, True, mask_2)

# Calculate the PyTorch attention result.
with sdpa_kernel(SDPBackend.MATH):
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_3)
    torch.cuda.synchronize()

# Create the total mask.
mask_4 = torch.where(mask == True, False, mask_2)

# Create the CSR mask.
mask_4 = mask_4.type(data_type)
csr_mask = mask_4.to_sparse_csr()
w_row_off = csr_mask.crow_indices().type(torch.uint64)
w_col_ind = csr_mask.col_indices().type(torch.uint32)
w_val = csr_mask.values()

# Create the buffers.
o_glob = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Run the local attention.
o_glob = spfa_local.forward(q, k, v, m, l, o_glob, WINDOW)
torch.cuda.synchronize()

# Run the global (non-local) attention.
o_glob = spfa_global_no_local.forward(q, k, v, m, l, o_glob, globs, WINDOW)
torch.cuda.synchronize()

# Run the random attention using CSR form.
o_glob = spfa_csr.forward(q, k, v, w_row_off, w_col_ind, w_val, m, l, o_glob, False)
torch.cuda.synchronize()

# Verify that we match the PyTorch output for BigBird attention.
if torch.allclose(torch_out, o_glob):
    print("BigBird (Multiple) attention matches PyTorch.")
else:
    print("FAIL")

# Print out the extrema differences.
print("Max difference from PyTorch (BigBird (Multiple) Attention):", torch.max(o_glob - torch_out))
print("Min difference from PyTorch (BigBird (Multiple) Attention):", torch.min(o_glob - torch_out))


# Local 1D Dilated Attention.
# Set the specific window and dilation parameters.
WINDOW = 10
DILATE = 2

# Create the mask.
mask_1d = torch.full((Q, Q), False, device = device, dtype = torch.bool)

for i in range(Q):
    start = max(i - (WINDOW * (DILATE + 1)), i % (DILATE + 1))
    end = min(i + (WINDOW * (DILATE + 1)), Q - 1)
    num_times = ((end - start) // (DILATE + 1)) + 1
    
    for j in range(num_times):
        mask_1d[i][start + (j * (DILATE + 1))] = True


# Calculate the PyTorch attention result.
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
with sdpa_kernel(SDPBackend.MATH):
# with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    torch_out_1d = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_1d)
    torch.cuda.synchronize()

# Set the new input tensors.
o_local_1d = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Run the SPFA-Local operation.
o_local_1d = spfa_local_1d_dilated.forward(q, k, v, m, l, o_local_1d, WINDOW, DILATE)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out_1d, o_local_1d, equal_nan = True):
    print("The SPFA-Local-1D-Dilated results match PyTorch's.")
else:
    print("FAIL")

# Print out the extrema differences.
print("Max difference from PyTorch (Local 1D Dilated Attention):", torch.max(o_local_1d - torch_out_1d))
print("Min difference from PyTorch (Local 1D Dilated Attention):", torch.min(o_local_1d - torch_out_1d))


# Local 2D Dilated Attention.
# Set the specific window and dilation parameters.
BLOCK = 16
DILATE = 0

# Create the mask.
mask_2d = torch.full((Q, Q), False, device = device, dtype = torch.bool)

for i in range(Q):
    block_num = i // BLOCK
    start = block_num * BLOCK

    count = min(BLOCK, ((BLOCK - 1) // (DILATE + 1)) + 1)

    if ((i - start) % (DILATE + 1) == 0):
        for j in range(count):
            mask_2d[i][start + (j * (DILATE + 1))] = True

# Calculate the PyTorch attention result.
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
with sdpa_kernel(SDPBackend.MATH):
# with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    torch_out_2d = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_2d)
    torch.cuda.synchronize()

# Set the new input tensors.
o_local_2d = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Run the SPFA-Local operation.
o_local_2d = spfa_local_2d_dilated.forward(q, k, v, m, l, o_local_2d, BLOCK, DILATE, True)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out_2d, o_local_2d, equal_nan = True):
    print("The SPFA-Local-2D-Dilated results match PyTorch's.")
else:
    print("FAIL")

# Print out the extrema differences.
print("Max difference from PyTorch (Local 2D Dilated Attention):", torch.max(o_local_2d - torch_out_2d))
print("Min difference from PyTorch (Local 2D Dilated Attention):", torch.min(o_local_2d - torch_out_2d))


# Local 2D Dilated Attention (w/NaN).
# Set the specific window and dilation parameters.
BLOCK = 16
DILATE = 1

# Create the mask.
mask_2d = torch.full((Q, Q), False, device = device, dtype = torch.bool)

for i in range(Q):
    block_num = i // BLOCK
    start = block_num * BLOCK

    count = min(BLOCK, ((BLOCK - 1) // (DILATE + 1)) + 1)

    if ((i - start) % (DILATE + 1) == 0):
        for j in range(count):
            mask_2d[i][start + (j * (DILATE + 1))] = True

# Calculate the PyTorch attention result.
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
with sdpa_kernel(SDPBackend.MATH):
# with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    torch_out_2d = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask_2d)
    torch.cuda.synchronize()

# Set the new input tensors.
o_local_2d = torch.zeros([Q, D], device = device, dtype = data_type)
m = torch.full([Q], -float("inf"), device = device, dtype = data_type)
l = torch.zeros(Q, device = device, dtype = data_type)

# Run the SPFA-Local operation.
o_local_2d = spfa_local_2d_dilated.forward(q, k, v, m, l, o_local_2d, BLOCK, DILATE, True)
torch.cuda.synchronize()

# Verify that the PyTorch SDPA's outputs are identical.
if torch.allclose(torch_out_2d, o_local_2d, equal_nan = True):
    print("The SPFA-Local-2D-Dilated w/NaN results match PyTorch's.")
else:
    print("FAIL")

# Print out the extrema differences.
print("Max difference from PyTorch (Local 2D Dilated Attention, w/NaN):", torch.max(o_local_2d - torch_out_2d))
print("Min difference from PyTorch (Local 2D Dilated Attention, w/NaN):", torch.min(o_local_2d - torch_out_2d))


# Finish the test runs.
print("Finished")