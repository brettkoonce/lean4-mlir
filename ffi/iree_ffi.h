// Public C API for the IREE FFI wrapper.
// See iree_ffi.c for implementation.
#ifndef IREE_FFI_H_
#define IREE_FFI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_ffi_session_t iree_ffi_session_t;

// Create a session from a .vmfb path. Returns NULL on failure.
iree_ffi_session_t* iree_ffi_session_create(const char* vmfb_path);

// Release a session (safe with NULL).
void iree_ffi_session_release(iree_ffi_session_t* sess);

// Invoke a function by name. All tensors float32.
// `input_ranks[i]` gives the rank of input i; `input_dims_flat` has the
// concatenated dimensions (sum of all ranks total entries).
// `output_totals[i]` is the expected element count for output i.
// Returns 0 on success, nonzero on failure (diagnostic printed to stderr).
int iree_ffi_invoke_f32(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int n_inputs,
    const int32_t* input_ranks,
    const int64_t* input_dims_flat,
    const float* const* input_data,
    int n_outputs,
    const int64_t* output_totals,
    float* const* output_data);

// Adam train step: pushes step counter t, pops BN stats after loss.
int iree_ffi_train_step_adam(
    iree_ffi_session_t* sess, const char* fn_name, int batch,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

// Adam train step for per-pixel segmentation. `y` is an int32
// [batch, H, W] per-pixel label tensor (instead of [batch] for
// classification). Routes to the codegen produced with
// `useSeg := true`.
int iree_ffi_train_step_adam_seg(
    iree_ffi_session_t* sess, const char* fn_name, int batch, int H, int W,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

#ifdef __cplusplus
}
#endif

#endif  // IREE_FFI_H_
