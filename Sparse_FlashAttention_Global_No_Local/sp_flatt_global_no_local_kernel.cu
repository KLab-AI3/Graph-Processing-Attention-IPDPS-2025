#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <stdexcept>

#include <torch/extension.h>


// The sparse-FlashAttention (local mask) forward CUDA kernel.
template <typename scalar_t>
__global__ void spfa_global_no_local_cuda_forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> V,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> m,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> l,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> O,
    const torch::PackedTensorAccessor64<int, 1, torch::RestrictPtrTraits> global_inds,
    const int local_size_each_dir,
    int maskElePerIter
) {
  // NOTE: I'm thinking that we have a coniditional up top that checks whether or not a blockIdx.x is in the global list.
  // We have a flag that will be set and dictates the path to take below.
  // utilize the indices below as a range of area that we do not calculate for this row
  // we set the total non mask to something else?
  // HOW DO WE DEAL WITH WHEN WE ARE IN THE INDICES? DO WE NOT ACCOUNT FOR THEM IN THE TOTAL NON MASKED?
  // account for the local mask terms, store the column indices in shared memory (only operated on by thread 0)

  // Create the shared memory object, aligned to the size of the template data type.
  extern __shared__ __align__(sizeof(scalar_t)) unsigned char sharedMem[];

  // Get pointers to the shared memory and reinterpret the pointer's data type.
  int* num_dot_prods = reinterpret_cast<int*>(sharedMem);

  // Calculate the upper and lower indices of the local mask.
  int lowInd = max((int)0, (int)(blockIdx.x - local_size_each_dir));
  int upInd = min((int)Q.size(0), (int)(blockIdx.x + local_size_each_dir + 1));

  // Define variables outside of the scope below, though they are solely used by thread 0.
  bool g_bool_flag = false;
  int dot_prods_done = 0;
  int offset = 0;

  // Find out whether or not this is a global row and the number of dot products required (adjusting for the local size).
  if (threadIdx.x == 0) {

    int included = 0;

    for (int i = 0; i < global_inds.size(0); i++) {
      int store = global_inds[i];

      if (store == blockIdx.x){
        included = Q.size(0) - (upInd - lowInd);
        g_bool_flag = true;
        break;
      } else if (store < lowInd || store > (upInd - 1)) {
        included += 1;
      }
    }

    num_dot_prods[0] = included;
  }

  // Block until all threads are synchronized after the last operation.
  __syncthreads();

  // Calculate the total number of non-masked elements in the row of the mask.
  int totNonMask = num_dot_prods[0];

  // Calculate the number of times the amount of threads will fit into the embedded dimension as well as the remainder.
  int intQuotient_ED = (int)(Q.size(1) / blockDim.x);
  int remainder_ED = (Q.size(1) % blockDim.x) - 1;

  // If the thread index is less than the remainder, then increment its quotient so it reads one more value.
  if (threadIdx.x <= remainder_ED) {
    intQuotient_ED += 1;
  }

  // If the total number of non-masked elements is under the max number of mask elements per iteration, 
  // then reduce the max number of elements for this iteration.
  if (totNonMask < maskElePerIter) {
    maskElePerIter = totNonMask;
  }

  // Get pointers to the shared memory and reinterpret the pointer's data type.
  scalar_t* Q_shared = (scalar_t*)&num_dot_prods[1];
  scalar_t* O_shared = (scalar_t*)&Q_shared[Q.size(1)];
  int* W_col_ind_shared = (int*)&O_shared[O.size(1)];
  scalar_t* W_val_shared = (scalar_t*)&W_col_ind_shared[maskElePerIter];
  scalar_t* K_shared = (scalar_t*)&W_val_shared[maskElePerIter];
  scalar_t* V_shared = (scalar_t*)&K_shared[maskElePerIter * K.size(1)];
  scalar_t* m_i = (scalar_t*)&V_shared[maskElePerIter * V.size(1)];
  scalar_t* m_ij = (scalar_t*)&m_i[1];
  scalar_t* m_new_i = (scalar_t*)&m_ij[1];
  scalar_t* l_i = (scalar_t*)&m_new_i[1];
  scalar_t* l_ij = (scalar_t*)&l_i[1];
  scalar_t* l_new_i = (scalar_t*)&l_ij[1];

  // Bring Q and O into shared memory.
  for (int i = 0; i < intQuotient_ED; i++) {
    Q_shared[(blockDim.x * i) + threadIdx.x] = Q[blockIdx.x][(blockDim.x * i) + threadIdx.x];
    O_shared[(blockDim.x * i) + threadIdx.x] = O[blockIdx.x][(blockDim.x * i) + threadIdx.x];
    // __syncthreads();
  }

  // Move the softmax statistics into shared memory.
  //// - m is initialized with 0.0.
  //// - l is initialized with -inf.
  //// - If they are set to something else in shared memory, then those values will be used.
  if (threadIdx.x == 0) {
    m_i[0] = m[blockIdx.x];
    l_i[0] = l[blockIdx.x];
  }

  __syncthreads();

  // Calculate the number of iterations that the block will have to perform.
  int blockIters = ceil(((float)totNonMask / (float)maskElePerIter));

  // Calculate the number of masked terms that remain on the last iteration.
  int numLastIter = (totNonMask % maskElePerIter);

  // The main calculation loop.
  for (int b_i = 0; b_i < blockIters; b_i++) {
    // If we're on the last iteration and it's not the first, then set the number of masked elements equal to those 
    // that have not been visited yet.
    if ((b_i == (blockIters - 1)) && (b_i != 0)) {
      maskElePerIter = numLastIter;
    }

    // Use thread 0 to map column (and row) indices in the shared column index memory.
    if (threadIdx.x == 0) {
      if (g_bool_flag == true) {
        for (int i = 0; i < maskElePerIter; i++) {
          int temp = i + dot_prods_done + offset;
          if (temp < lowInd || temp > (upInd - 1)) {
            W_col_ind_shared[i] = temp;
          } else {
            temp += (upInd - lowInd);
            W_col_ind_shared[i] = temp;
            offset += (upInd - lowInd);
          }
        }
      } else {
        for (int i = 0; i < maskElePerIter; i++) {
          int temp = global_inds[i + dot_prods_done + offset];
          if (temp < lowInd || temp > (upInd - 1)) {
            W_col_ind_shared[i] = temp;
          } else {
            for (int j = dot_prods_done + i + offset + 1; j < global_inds.size(0); j++) {
              temp = global_inds[j];
              offset += 1;
              if (temp < lowInd || temp > (upInd - 1)) {
                break;
              }
            }
            W_col_ind_shared[i] = temp;
          }
        }
      }

      dot_prods_done += maskElePerIter;

    }

    __syncthreads();

    // Calculate the rows and columns of K and V and bring them into shared memory.
    //// - W's values are overwritten each iteration, so there is no need to initialize it to something.
    //// - K_shared is contiguous along the rows of K.
    //// - V_shared is contiguous along the columns of V.
    if (threadIdx.x < maskElePerIter) {
      // Store the column index of the mask so we don't have to get it each iteration below.
      int col_ind = W_col_ind_shared[threadIdx.x];

      for (int i = 0; i < K.size(1); i++) {
        K_shared[(K.size(1) * threadIdx.x) + i] = K[col_ind][i];
        V_shared[(maskElePerIter * i) + threadIdx.x] = V[col_ind][i];
      }
    }

    __syncthreads();

    if (threadIdx.x < maskElePerIter) {
      scalar_t acc = 0.0;

      for (int i = 0; i < Q.size(1); i++) {
        acc += Q_shared[i] * K_shared[(Q.size(1) * threadIdx.x) + i];
      }

      W_val_shared[threadIdx.x] = acc / sqrt((float)Q.size(1));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      scalar_t row_max = - (1.0 / 0.0);

      for (int i = 0; i < maskElePerIter; i++) {
        if (W_val_shared[i] > row_max) {
          row_max = W_val_shared[i];
        }
      }

      m_ij[0] = row_max;
    }

    __syncthreads();

    if (threadIdx.x < maskElePerIter) {
      W_val_shared[threadIdx.x] = exp(W_val_shared[threadIdx.x] - m_ij[0]);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      scalar_t row_sum = 0.0;

      for (int i = 0; i < maskElePerIter; i++) {
        row_sum += W_val_shared[i];
      }

      l_ij[0] = row_sum;

      if (m_i[0] > m_ij[0]) {
        m_new_i[0] = m_i[0];
      }
      else {
        m_new_i[0] = m_ij[0];
      }

      l_new_i[0] = (l_i[0] * exp(m_i[0] - m_new_i[0])) + (l_ij[0] * exp(m_ij[0] - m_new_i[0]));
    }

    __syncthreads();

    // NOTE: This does not work for sizes greater than number of threads.
    if (threadIdx.x < V.size(1)) {
      scalar_t acc = 0.0;

      for (int i = 0; i < maskElePerIter; i++) {
        acc += W_val_shared[i] * V_shared[(threadIdx.x * maskElePerIter) + i];
      }

      O_shared[threadIdx.x] = (1.0 / l_new_i[0]) * 
        ((l_i[0] * O_shared[threadIdx.x] * exp(m_i[0] - m_new_i[0])) + 
        (acc * exp(m_ij[0] - m_new_i[0])));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      m_i[0] = m_new_i[0];
      l_i[0] = l_new_i[0];
    }

    __syncthreads();

  }

  // Update the softmax statistics in HBM.
  if (threadIdx.x == 0) {
      m[blockIdx.x] = m_new_i[0];
      l[blockIdx.x] = l_new_i[0];
  }

  __syncthreads();

  for (int i = 0; i < intQuotient_ED; i++) {
    O[blockIdx.x][(blockDim.x * i) + threadIdx.x] = O_shared[(blockDim.x * i) + threadIdx.x];
    // __syncthreads();
  }
  
}


// The CUDA kernel dispatch interface for the sparse-FlashAttention (local mask) forward CUDA kernel.
torch::Tensor spfa_global_no_local_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor m,
    torch::Tensor l,
    torch::Tensor O,
    torch::Tensor global_inds,
    int local_size_each_dir
) {
  // Initialize the variables to hold the desired device attributes.
  //// - Max number of threads per block.
  //// - Max shared memory per block (bytes).
  int maxThrPB, maxSMemPB;

  // Get the device attributes.
  cudaDeviceGetAttribute(&maxThrPB, cudaDevAttrMaxThreadsPerBlock, 0);
  cudaDeviceGetAttribute(&maxSMemPB, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  // Set the grid dimensions for the problem.
  //// - The number of threads is set to the max per block for the device.
  //// - The number of blocks is equal to the number of rows in Q.
  const int threads = maxThrPB;
  const dim3 blocks(Q.size(0));

  // Store the embedded dimension.
  int embD = Q.size(1);

  // Calculate the max number of mask elements that will be utilized for each iteration of building O.
  //// - maxSMemPB -> The amount of shared memory available to a block in bytes.
  //// - (2 * embD * sizeof(Q.type())) -> (numerator) The amount of shared memory that storing Q and O will require.
  //// - (6 * sizeof(l.type())) -> The amount of shared memory that storing the softmax statistics will require.
  //// - (sizeof(int)) -> The amount of shared memory required in order to store the number of dot products that will be done.
  //// - sizeof(Q.type()) -> The amount of shared memory that storing the mask's temporary values will require.
  //// - sizeof(int) -> The amount of shared memory that storing the mask's global indices will require.
  //// - (2 * embD * sizeof(Q.type())) -> (denominator) The amount of shared memory that storing K and V will require.
  int maskElePerIter = floor((maxSMemPB - (2 * embD * sizeof(Q.type())) - (6 * sizeof(l.type())) - (sizeof(int))) / 
    (sizeof(Q.type()) + sizeof(int) + (2 * embD * sizeof(Q.type()))));

  // If the number of mask elements per iteration is less than 1, then throw an error.
  if (maskElePerIter < 1) {
    throw std::invalid_argument( "The embedded dimension is too large for the amount of shared memory on your GPU." );
  }

  // Throw an error if no global tokens, or too many (greater than the sequence length), are provided.
  if (global_inds.size(0) == 0 || global_inds.size(0) > Q.size(0)) {
    throw std::invalid_argument( "There are either no global tokens provided or there are more than the sequence length given." );
  }

  // The local size parameter specifies the numer of terms adjacent to each direction from the identity diagonal. 
  // If the specified size is negative or is greater than (1 - Sequence Length, resulting in a local region greather than 
  // the tensor's size (symmetric)), then raise an error.
  if (local_size_each_dir < 0 || local_size_each_dir > (Q.size(0) - 1)) {
    throw std::invalid_argument( "The local size specified is either negative or greater than/equal to (1 - Sequence Length)." );
  }

  // Dispatcher that handles launching the correctly typed function from the generic implementation.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.type(), "spfa_global_no_local_cuda_forward", ([&] {
    spfa_global_no_local_cuda_forward_kernel<scalar_t><<<blocks, threads, maxSMemPB>>>(
        Q.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        K.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        V.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        m.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        l.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        O.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        global_inds.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
        local_size_each_dir,
        maskElePerIter
      );
  }));

  // Return the output tensor.
  return O;
}