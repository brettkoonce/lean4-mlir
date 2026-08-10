// Runtime lowerer dispatch: the shim surface, resolved by dlopen instead of at link.
//
// WHY THIS EXISTS
// ---------------
// `libiree_ffi.so` and `libpjrt_ffi.so` export the SAME ten `iree_ffi_*` symbols
// (`nm -D`), so which lowerer a binary uses was decided by the link line: two
// `moreLinkArgs` in the lakefile, hence two executables per demo, hence a third
// `*Common.lean` per demo to stop their configs drifting. "The program" ended up
// spread across three files.
//
// The seam was already the right one and already dynamic in spirit -- these ten
// were plain `extern` declarations resolved from whichever `.so` was named. This
// header moves that resolution to run time. Each name becomes a function POINTER
// with the same name (C calls through them with identical syntax), so no call site
// in `iree_lean_ffi.c` changes.
//
// One binary per demo, `$LEAN_MLIR_LOWERER=xla|iree`, one printable Lean file.
//
// THE `pjrt_ffi_marker` INVARIANT IS PRESERVED. `lean_iree_backend_name` reports
// the backend by testing that symbol, deliberately rather than reading an env var
// "that could disagree with the binary". It still cannot disagree: the marker is
// non-NULL iff the PJRT shim is what actually dlopened. It reports what loaded,
// not what was requested.
#ifndef LOWERER_H_
#define LOWERER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_ffi_session_t iree_ffi_session_t;

// The one real function, not a pointer: it performs the one-time dlopen before
// forwarding. Nothing else in the surface is reachable without a session, so by
// the time any pointer below is called it is guaranteed resolved. That is why
// none of them need a NULL check or a stub.
iree_ffi_session_t* lowerer_session_create(const char* path);

// Names the loaded lowerer: "xla", "iree", or "none" before the first session.
const char* lowerer_active_name(void);

// ---- the shared surface: present in BOTH shims ----
extern void (*lowerer_session_release)(iree_ffi_session_t*);

extern int (*lowerer_invoke_f32)(
    iree_ffi_session_t*, const char* fn_name, int n_inputs,
    const int32_t* input_ranks, const int64_t* input_dims_flat,
    const float* const* input_data,
    int n_outputs, const int64_t* output_totals, float* const* output_data);

extern int (*lowerer_train_step_mlp)(
    iree_ffi_session_t*, const char* fn_name, int batch,
    const float* W0, const float* b0, const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* x, const int32_t* y, float lr,
    float* W0_new, float* b0_new, float* W1_new, float* b1_new,
    float* W2_new, float* b2_new, float* loss_out);

extern int (*lowerer_train_step_generic)(
    iree_ffi_session_t*, const char* fn_name, int batch, int n_params,
    const int32_t* param_ranks, const int64_t* param_dims_flat,
    const int64_t* param_sizes, const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr,
    float* packed_params_out, float* loss_out);

extern int (*lowerer_train_step_adam)(
    iree_ffi_session_t*, const char* fn_name, int batch, int n_params,
    const int32_t* param_ranks, const int64_t* param_dims_flat,
    const int64_t* param_sizes, const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

extern int (*lowerer_train_step_adam_seg)(
    iree_ffi_session_t*, const char* fn_name, int batch, int H, int W,
    int n_params,
    const int32_t* param_ranks, const int64_t* param_dims_flat,
    const int64_t* param_sizes, const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

extern int (*lowerer_train_step_adam_softlabel)(
    iree_ffi_session_t*, const char* fn_name, int batch, int n_classes,
    int n_params,
    const int32_t* param_ranks, const int64_t* param_dims_flat,
    const int64_t* param_sizes, const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const float* y_soft, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

extern int (*lowerer_train_step_adam_ddpm)(
    iree_ffi_session_t*, const char* fn_name, int batch,
    int outC, int outH, int outW, int n_params,
    const int32_t* param_ranks, const int64_t* param_dims_flat,
    const int64_t* param_sizes, const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const float* y_ddpm, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

extern int (*lowerer_train_step_adam_yolov1)(
    iree_ffi_session_t*, const char* fn_name, int batch,
    int gridH, int gridW, int perCell, int n_params,
    const int32_t* param_ranks, const int64_t* param_dims_flat,
    const int64_t* param_sizes, const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const float* y_yolo, const float* m_yolo, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

// ---- XLA-only extras: NULL when the IREE shim is loaded ----
// These were `__attribute__((weak))` externs, i.e. "NULL unless the PJRT shim is
// on the link line". dlsym-returns-NULL is the same contract, so every existing
// `if (pjrt_ffi_x)` guard keeps working unchanged.
extern void (*lowerer_pjrt_marker)(void);

extern int (*lowerer_pjrt_invoke_f32_resident_v2)(
    iree_ffi_session_t*, const char*, int, int, int, int, long long, int,
    const int32_t*, const int64_t*, const float* const*,
    const unsigned char*, int, const int64_t*, float* const*);

extern int (*lowerer_pjrt_resident_read)(iree_ffi_session_t*, int64_t, float*);

extern int (*lowerer_pjrt_invoke_f32_dp)(
    iree_ffi_session_t*, const char*, int, int,
    const int32_t*, const int64_t*, const float* const*,
    const unsigned char*, int, const int64_t*, float* const*);

// Keep every call site in iree_lean_ffi.c spelled exactly as before.
#define iree_ffi_session_create            lowerer_session_create
#define iree_ffi_session_release           lowerer_session_release
#define iree_ffi_invoke_f32                lowerer_invoke_f32
#define iree_ffi_train_step_mlp            lowerer_train_step_mlp
#define iree_ffi_train_step_generic        lowerer_train_step_generic
#define iree_ffi_train_step_adam           lowerer_train_step_adam
#define iree_ffi_train_step_adam_seg       lowerer_train_step_adam_seg
#define iree_ffi_train_step_adam_softlabel lowerer_train_step_adam_softlabel
#define iree_ffi_train_step_adam_ddpm      lowerer_train_step_adam_ddpm
#define iree_ffi_train_step_adam_yolov1    lowerer_train_step_adam_yolov1
#define pjrt_ffi_marker                    lowerer_pjrt_marker
#define pjrt_ffi_invoke_f32_resident_v2    lowerer_pjrt_invoke_f32_resident_v2
#define pjrt_ffi_resident_read             lowerer_pjrt_resident_read
#define pjrt_ffi_invoke_f32_dp             lowerer_pjrt_invoke_f32_dp

#ifdef __cplusplus
}
#endif

#endif  // LOWERER_H_
