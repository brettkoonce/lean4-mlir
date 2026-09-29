// Lean FFI shim for the IREE runtime wrapper.
// Converts between Lean's FloatArray (Float64) and IREE's expected float32.
//
// Exports: the `@[extern]` bodies of LeanMlir/IreeRuntime.lean, one per opaque.

#include <lean/lean.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
// The shim surface is resolved by dlopen at run time, not from the link line --
// see ffi/lowerer.h. Every `iree_ffi_*` / `pjrt_ffi_*` name below is a function
// pointer of the same name, so all call sites in this file are unchanged.
#include "lowerer.h"

// ---- Which backend shim is loaded? ----
// `libiree_ffi.so` and `libpjrt_ffi.so` export the same symbols; only the latter
// defines `pjrt_ffi_marker`. It is now NULL-or-not by dlsym rather than by weak
// linkage, which keeps the property this check was written for: it reports what
// ACTUALLY loaded, so it still cannot disagree with the running program the way
// a bare env var could. See planning/archive/xla_pjrt_ladder.md.

LEAN_EXPORT lean_obj_res lean_iree_backend_name(void) {
  // `lowerer_active_name()` rather than a bare `pjrt_ffi_marker` test: this is
  // the FIRST thing the driver asks, before any session exists, so it has to be
  // what triggers the dlopen. It still answers with what actually loaded --
  // internally it is the same marker check, just no longer able to run early.
  return lean_io_result_mk_ok(lean_mk_string(lowerer_active_name()));
}

// ---- Device-resident parameters (handoff §2d.3) ----
// Both entry points are exported ONLY by the XLA shim, so the references are
// NULL-or-not by dlsym exactly as `pjrt_ffi_invoke_f32_dp`'s is: the IREE build
// links fine and takes the copying path unconditionally.
//
// ⚠ THE SWITCH LIVES IN C, AND THAT IS DELIBERATE. The FFI surface is
// symbol-identical across the two shims by design (`nm -D`), and every
// cross-backend gate in the repo depends on IREE and XLA running the SAME Lean
// code path — a backend branch inside the training loop would break the "one
// shared body, cannot drift" property those gates are built on (§2d.3, "the
// design decision that protects every existing gate"). So the Lean driver
// always calls the same function with the same arguments; this file decides
// which transport serves it, and on IREE the question never arises.
//
// The switch itself is read ONCE, inside the XLA shim (`resident_enabled` in
// pjrt_ffi.c: on unless `$PJRT_FFI_RESIDENT=0`, since 2026-09-29), and asked
// for here through `pjrt_ffi_resident_available` — NULL on IREE, which is "off".
// Reading the env var here as well is how two copies of one rule drift.
static int resident_wanted(void) {
  return pjrt_ffi_resident_available ? pjrt_ffi_resident_available() : 0;
}

// `n_resident` is a tensor COUNT supplied by the driver, which is the only place
// that knows the packed layout is `[theta|m|v | lr,bc1,bc2 | bn stats]` and
// therefore that the leading 3xP tensors are the ones the host never reads back.
// Zero — the default for every call site that has not opted in — means "copying
// path", so the tie and DP-check harnesses, which DO read the whole output, are
// unaffected by construction.
static int use_resident(size_t n_resident) {
  return n_resident > 0 && resident_wanted() && pjrt_ffi_invoke_f32_resident_v2 != NULL;
}

// ---- External class for IreeSession ----
static lean_external_class* g_iree_session_class = NULL;

static void iree_session_finalize(void* p) {
  iree_ffi_session_release((iree_ffi_session_t*)p);
}
static void iree_session_foreach(void* p, b_lean_obj_arg f) { (void)p; (void)f; }

static void ensure_iree_session_class(void) {
  if (!g_iree_session_class) {
    g_iree_session_class = lean_register_external_class(
        iree_session_finalize, iree_session_foreach);
  }
}

// ---- Session create ----
LEAN_EXPORT lean_obj_res lean_iree_session_create(
    b_lean_obj_arg path_obj) {
  ensure_iree_session_class();
  const char* path = lean_string_cstr(path_obj);
  iree_ffi_session_t* sess = iree_ffi_session_create(path);
  if (!sess) {
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string("iree_ffi_session_create failed (see stderr)")));
  }
  return lean_io_result_mk_ok(
      lean_alloc_external(g_iree_session_class, sess));
}

// ---- Session create, SHARDED INFERENCE (XLA only) ----
// The eval forward compiled for `replicas` devices with its outputs gathered from
// all of them — see `pjrt_ffi_session_create_dp`. Drive it with
// `lean_iree_forward_f32_dp` at the same replica count.
LEAN_EXPORT lean_obj_res lean_iree_session_create_dp(
    b_lean_obj_arg path_obj, size_t replicas) {
  ensure_iree_session_class();
  const char* path = lean_string_cstr(path_obj);
  iree_ffi_session_t* sess = lowerer_session_create_dp(path, (int)replicas);
  if (!sess) {
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string("pjrt_ffi_session_create_dp failed (see stderr)")));
  }
  return lean_io_result_mk_ok(
      lean_alloc_external(g_iree_session_class, sess));
}

// ---- Adam train step (f32, with step counter t + BN stats output) ----
// bnShapes: packed int32 array [n_bn_layers, oc0, oc1, ...] — each oc appears twice (mean + var)
// Returns: ByteArray of (total_params + 1 + total_bn_stats) floats
LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Parse param shapes
  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  // Parse BN shapes: [n_bn_layers, oc0, oc1, ...]
  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;      // mean size
      bn_sizes[i * 2 + 1] = oc;  // var size
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_ba);

  // Output: params + loss + bn_stats
  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc = iree_ffi_train_step_adam(
      sess, fn_name, (int)batch,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_ptr, (float)lr, (float)t,
      rp, &loss_f,
      n_bn_layers, bn_sizes, bn_out);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

// ---- Soft-label train step: y_soft is [batch, n_classes] f32 ----
LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32_softlabel(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_soft_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t n_classes) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;
      bn_sizes[i * 2 + 1] = oc;
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const float* y_soft = (const float*)lean_sarray_cptr(y_soft_ba);

  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc = iree_ffi_train_step_adam_softlabel(
      sess, fn_name, (int)batch, (int)n_classes,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_soft, (float)lr, (float)t,
      rp, &loss_f,
      n_bn_layers, bn_sizes, bn_out);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 softlabel train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

// ---- Adam train step (f32, dense f32 target: DDPM noise, DQN Bellman targets) ----
// `y_ddpm_ba` is an f32 [batch, outC, outH, outW] target.
// Routes to the codegen produced with `useDdpm := true`.
//
// `n_resident` > 0 on the XLA shim (resident unless PJRT_FFI_RESIDENT=0) keeps the leading
// `n_resident` param tensors (`[theta|m|v]`, inputs AND outputs 0..np-1 — the graph's
// params-in / params-out correspondence is index for index) on the device, as
// `lean_iree_linear_train_step` does; the result's param region is then unwritten
// and `lean_iree_read_params[_prefix]` is the way back. Zero, or any other backend,
// is the copying path through `iree_ffi_train_step_adam_ddpm`, unchanged.
static lean_obj_res train_step_adam_f32_ddpm_core(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_ddpm_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t outC, size_t outH, size_t outW, size_t n_resident) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;
      bn_sizes[i * 2 + 1] = oc;
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const float* y_ptr = (const float*)lean_sarray_cptr(y_ddpm_ba);

  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc;
  if (use_resident(n_resident) && (int)n_resident == n_params) {
    // The shim's `iree_ffi_train_step_adam_ddpm` layout, spelled here so the resident
    // invoke can take it: inputs params, x, y [b,oC,oH,oW], lr, t (rank 0); outputs
    // params, loss, then the BN stat pairs.
    const int n_inputs = n_params + 4;
    const int n_outputs = n_params + 1 + n_bn_layers * 2;
    int32_t* ranks = (int32_t*)malloc((size_t)n_inputs * sizeof(int32_t));
    int64_t* dims = (int64_t*)malloc((size_t)(dims_idx + x_rank + 4 + 1) * sizeof(int64_t));
    const float** ins = (const float**)malloc((size_t)n_inputs * sizeof(float*));
    int64_t* totes = (int64_t*)malloc((size_t)n_outputs * sizeof(int64_t));
    float** outs = (float**)malloc((size_t)n_outputs * sizeof(float*));
    float lr_v = (float)lr, t_v = (float)t;
    int di = 0;
    int64_t off = 0;
    for (int i = 0; i < n_params; i++) {
      ranks[i] = param_ranks[i];
      for (int k = 0; k < param_ranks[i]; k++) { dims[di] = param_dims_flat[di]; di++; }
      ins[i] = p_f + off;
      totes[i] = param_sizes[i];
      outs[i] = rp + off;
      off += param_sizes[i];
    }
    ranks[n_params] = x_rank;
    for (int k = 0; k < x_rank; k++) dims[di++] = x_dims[k];
    ins[n_params] = x_f;
    ranks[n_params + 1] = 4;
    dims[di++] = (int64_t)batch; dims[di++] = (int64_t)outC;
    dims[di++] = (int64_t)outH;  dims[di++] = (int64_t)outW;
    ins[n_params + 1] = y_ptr;
    ranks[n_params + 2] = 0; ins[n_params + 2] = &lr_v;
    ranks[n_params + 3] = 0; ins[n_params + 3] = &t_v;
    totes[n_params] = 1; outs[n_params] = &loss_f;
    int64_t boff = 0;
    for (int i = 0; i < n_bn_layers * 2; i++) {
      totes[n_params + 1 + i] = bn_sizes[i];
      outs[n_params + 1 + i] = bn_out + boff;
      boff += bn_sizes[i];
    }
    rc = pjrt_ffi_invoke_f32_resident_v2(sess, fn_name, 1,
        /*res_in=*/0, /*res_out=*/0, (int)n_resident, /*res_gen=*/0,
        n_inputs, ranks, dims, ins, NULL,
        n_outputs, totes, outs);
    free(ranks); free(dims); free(ins); free(totes); free(outs);
  } else {
    rc = iree_ffi_train_step_adam_ddpm(
        sess, fn_name, (int)batch, (int)outC, (int)outH, (int)outW,
        n_params, param_ranks, param_dims_flat, param_sizes,
        p_f, x_rank, x_dims, x_f, y_ptr, (float)lr, (float)t,
        rp, &loss_f,
        n_bn_layers, bn_sizes, bn_out);
  }

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 ddpm train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32_ddpm(
    b_lean_obj_arg sess_obj, b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba, b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba, b_lean_obj_arg x_shape_ba, b_lean_obj_arg y_ddpm_ba,
    double lr, double t, b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t outC, size_t outH, size_t outW) {
  return train_step_adam_f32_ddpm_core(sess_obj, fn_name_obj, params_ba, shapes_ba,
      x_ba, x_shape_ba, y_ddpm_ba, lr, t, bn_shapes_ba, batch, outC, outH, outW, 0);
}

LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32_ddpm_r(
    b_lean_obj_arg sess_obj, b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba, b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba, b_lean_obj_arg x_shape_ba, b_lean_obj_arg y_ddpm_ba,
    double lr, double t, b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t outC, size_t outH, size_t outW, size_t n_resident) {
  return train_step_adam_f32_ddpm_core(sess_obj, fn_name_obj, params_ba, shapes_ba,
      x_ba, x_shape_ba, y_ddpm_ba, lr, t, bn_shapes_ba, batch, outC, outH, outW, n_resident);
}

// YOLOv1 variant. y_yolo is f32 [batch, perCell, gridH, gridW] (target);
// m_yolo is f32 [batch, gridH, gridW] (per-cell objectness mask). Routes
// to the codegen produced with `useYolov1 := true`. See
// planning/archive/yolo_demo_v2.md Phase 1 decisions D3 + D6.
LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32_yolov1(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_yolo_ba,
    b_lean_obj_arg m_yolo_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t gridH, size_t gridW, size_t perCell) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;
      bn_sizes[i * 2 + 1] = oc;
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const float* y_ptr = (const float*)lean_sarray_cptr(y_yolo_ba);
  const float* m_ptr = (const float*)lean_sarray_cptr(m_yolo_ba);

  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc = iree_ffi_train_step_adam_yolov1(
      sess, fn_name, (int)batch, (int)gridH, (int)gridW, (int)perCell,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_ptr, m_ptr, (float)lr, (float)t,
      rp, &loss_f,
      n_bn_layers, bn_sizes, bn_out);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 yolov1 train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32_seg(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_seg_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t H, size_t W) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;
      bn_sizes[i * 2 + 1] = oc;
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_seg_ba);

  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc = iree_ffi_train_step_adam_seg(
      sess, fn_name, (int)batch, (int)H, (int)W,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_ptr, (float)lr, (float)t,
      rp, &loss_f,
      n_bn_layers, bn_sizes, bn_out);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 seg train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

// ---- Zero-copy f32 generic forward pass ----
// Pushes x first, then param tensors. Returns logits as ByteArray.
// Forward signature: forward(x, W0, g0, bt0, W1, ...) -> logits
//
// ONE body behind both exports, the `invoke_typed` idiom: `replicas == 1` is
// `lean_iree_forward_f32` and makes exactly the calls it always made;
// `replicas > 1` is `lean_iree_forward_f32_dp` — x (input 0) sharded by rows,
// the parameters replicated, the logits gathered back in row order by the shim.
// `batch` is then the GLOBAL batch, `replicas` × the rendered one.
static lean_obj_res forward_core(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    size_t batch, size_t n_classes, size_t n_resident, size_t res_gen,
    size_t replicas) {
  if (replicas > 1 && !pjrt_ffi_invoke_f32_dp) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
        "sharded forward needs the XLA shim (libpjrt_ffi.so)")));
  }
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Parse param shapes
  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int sp_idx = 1;
  int64_t total_params = 0;

  // Count total inputs: 1 (x) + n_params
  int n_inputs = 1 + n_params;
  int32_t* input_ranks = (int32_t*)malloc(n_inputs * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4 + 8;
  int64_t* input_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  const float** input_data = (const float**)malloc(n_inputs * sizeof(float*));

  // First input: x
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  input_ranks[0] = x_rank;
  int dims_idx = 0;
  for (int i = 0; i < x_rank; i++)
    input_dims_flat[dims_idx++] = (int64_t)xsp[1+i];
  input_data[0] = (const float*)lean_sarray_cptr(x_ba);

  // Remaining inputs: param tensors
  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  int64_t data_off = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    input_ranks[1+i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      input_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= input_dims_flat[dims_idx];
      dims_idx++;
    }
    input_data[1+i] = p_f + data_off;
    data_off += sz;
    total_params += sz;
  }

  // Output: logits (batch x n_classes)
  int64_t logits_total = (int64_t)(batch * n_classes);
  size_t out_bytes = logits_total * 4;
  lean_object* result = lean_alloc_sarray(1, out_bytes, out_bytes);
  float* logits = (float*)lean_sarray_cptr(result);

  int64_t out_totals[1] = {logits_total};
  float* outputs[1] = {logits};

  // Residency, HOLD mode (§2d.3). `res_out = -1`: this graph returns logits, not
  // parameters, so there is nothing to retain from the output — the parameters are
  // seeded once and reused across every eval batch instead of being pushed 79-123
  // times per epoch. Measured on the MNIST MLP: **73% of an eval step** was the
  // param push (0.6 ms of 0.8; compute is 0.1).
  //
  // `res_gen` is the caller's generation token and is what makes this safe: the
  // parameters change once per epoch, the caller changes the token, and the shim
  // re-seeds. A held set that went stale silently would score last epoch's weights
  // and read as a training plateau rather than as an error.
  int rc;
  if (replicas <= 1) {
    rc = use_resident(n_resident)
      ? pjrt_ffi_invoke_f32_resident_v2(sess, fn_name, 1,
          /*res_in=*/1, /*res_out=*/-1, (int)n_resident, (long long)res_gen,
          n_inputs, input_ranks, input_dims_flat, input_data, NULL,
          1, out_totals, outputs)
      : iree_ffi_invoke_f32(sess, fn_name,
          n_inputs, input_ranks, input_dims_flat, input_data,
          1, out_totals, outputs);
  } else {
    // Sharded: x only. Hold mode then keeps one parameter set per DEVICE, seeded
    // once per `res_gen` exactly as above, so the parameter push is N× once per
    // epoch — not N× per batch, which is what the copying `_dp` path would cost.
    unsigned char* shard = (unsigned char*)calloc((size_t)n_inputs, 1);
    shard[0] = 1;
    rc = use_resident(n_resident)
      ? pjrt_ffi_invoke_f32_resident_v2(sess, fn_name, (int)replicas,
          /*res_in=*/1, /*res_out=*/-1, (int)n_resident, (long long)res_gen,
          n_inputs, input_ranks, input_dims_flat, input_data, shard,
          1, out_totals, outputs)
      : pjrt_ffi_invoke_f32_dp(sess, fn_name, (int)replicas,
          n_inputs, input_ranks, input_dims_flat, input_data, shard,
          1, out_totals, outputs);
    free(shard);
  }

  free(input_ranks); free(input_dims_flat); free(input_data);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string(
            replicas > 1 ? "sharded f32 forward failed" : "f32 forward failed")));
  }
  return lean_io_result_mk_ok(result);
}

LEAN_EXPORT lean_obj_res lean_iree_forward_f32(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    size_t batch, size_t n_classes, size_t n_resident, size_t res_gen) {
  return forward_core(sess_obj, fn_name_obj, params_ba, shapes_ba, x_ba, x_shape_ba,
                      batch, n_classes, n_resident, res_gen, 1);
}

// The sharded eval forward. `sess` must come from `lean_iree_session_create_dp`
// at this same `replicas` — the shim refuses a count the executable was not
// compiled for.
LEAN_EXPORT lean_obj_res lean_iree_forward_f32_dp(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    size_t batch, size_t n_classes, size_t replicas,
    size_t n_resident, size_t res_gen) {
  return forward_core(sess_obj, fn_name_obj, params_ba, shapes_ba, x_ba, x_shape_ba,
                      batch, n_classes, n_resident, res_gen, replicas);
}

// ---- Verified-renderer linear train step (StableHLO.linearTrainStepModuleV) ----
// Drives the proof-rendered @linear_train_step (signature
//   (x:[B,d0], W0:[d0,d1], b0:[d1], onehot:[B,d1]) -> (W0n:[d0,d1], b0n:[d1]))
// through the generic IREE invoke. The one-hot is built here from int32 labels
// `y` (so the Lean caller passes the same labels the production path uses).
// Returns a ByteArray of W0n (d0*d1 f32) ++ b0n (d1 f32).
// ── Target construction: ONE definition, shared by all three train-step entry points ──────────
//
// Every render takes the target as a `[batch, nClasses]` FLOAT tensor (`%onehot`), never as
// integer labels — so the graph has always been able to consume an arbitrary target
// distribution. What forced hard labels was this C layer expanding int32s into a one-hot, in
// three separate copies. This is that expansion, once, with a soft path beside it.
//
// **The buffer is SELF-DESCRIBING BY SIZE**, deliberately, rather than gated by a new flag
// argument:
//   * `batch * 4` bytes           → int32 hard labels, expanded to a one-hot (the old behaviour,
//                                   bit-for-bit — every existing caller lands here unchanged);
//   * `batch * nClasses * 4` bytes → float32 target distribution, copied through. This is what
//                                   mixup/cutmix produce (`λ·y_a + (1−λ)·y_b`), and what BCE-style
//                                   multi-hot targets would use.
//   * anything else                → REFUSED, loudly.
//
// Why size and not a flag: a flag is a signature change on three `@[extern]` entry points, and
// §2d.3 paid for exactly that lesson — a stale binary calling a changed signature shifts every
// argument and produces garbage rather than a link error. Size dispatch cannot be stale: the two
// sizes differ by a factor of `nClasses`, which is ≥ 2 for every classification net in the repo,
// so they are never ambiguous. The `d3 > 1` guard makes that precondition explicit instead of
// assumed, and the else-branch refuses rather than guessing.
//
// ⚠ Nothing here validates that a soft target is a DISTRIBUTION (non-negative, sums to 1). It is
// not this layer's job — BCE targets are multi-hot and legitimately do not sum to 1 — but it does
// mean a caller that ships garbage gets a silently wrong loss, so the producer needs its own gate.
//
// Returns 0 on success, -1 if the buffer size matches neither convention.
static int lean_fill_targets(b_lean_obj_arg y_ba, size_t batch, size_t nclasses, float* out) {
  const size_t nbytes = lean_sarray_size(y_ba);
  const size_t hard_bytes = batch * sizeof(int32_t);
  const size_t soft_bytes = batch * nclasses * sizeof(float);
  if (nclasses > 1 && nbytes == soft_bytes) {
    memcpy(out, lean_sarray_cptr(y_ba), soft_bytes);
    return 0;
  }
  if (nbytes == hard_bytes) {
    const int32_t* y = (const int32_t*)lean_sarray_cptr(y_ba);
    memset(out, 0, batch * nclasses * sizeof(float));
    for (size_t i = 0; i < batch; i++) {
      int32_t l = y[i];
      if (l >= 0 && (size_t)l < nclasses) out[i * nclasses + (size_t)l] = 1.0f;
    }
    return 0;
  }
  return -1;
}

LEAN_EXPORT lean_obj_res lean_iree_linear_train_step(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg w0_ba,
    b_lean_obj_arg b0_ba,
    b_lean_obj_arg y_ba,
    size_t batch, size_t d0, size_t d1, size_t n_resident) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Target [batch, d1] f32 — int32 hard labels or a float32 distribution, see lean_fill_targets.
  float* onehot = (float*)calloc(batch * d1, sizeof(float));
  if (lean_fill_targets(y_ba, batch, d1, onehot) != 0) {
    free(onehot);
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
        "linear train step: target buffer is neither int32[batch] nor float32[batch*nClasses]")));
  }

  // 4 inputs: x[B,d0], W0[d0,d1], b0[d1], onehot[B,d1].
  int32_t input_ranks[4]   = {2, 2, 1, 2};
  int64_t input_dims_flat[7] = {(int64_t)batch, (int64_t)d0,
                                (int64_t)d0,    (int64_t)d1,
                                (int64_t)d1,
                                (int64_t)batch, (int64_t)d1};
  const float* input_data[4] = {
      (const float*)lean_sarray_cptr(x_ba),
      (const float*)lean_sarray_cptr(w0_ba),
      (const float*)lean_sarray_cptr(b0_ba),
      (const float*)onehot};

  // 2 outputs packed into one result: W0n (d0*d1) ++ b0n (d1).
  int64_t n_w = (int64_t)(d0 * d1), n_b = (int64_t)d1;
  size_t out_bytes = (size_t)(n_w + n_b) * 4;
  lean_object* result = lean_alloc_sarray(1, out_bytes, out_bytes);
  float* out = (float*)lean_sarray_cptr(result);
  int64_t out_totals[2] = {n_w, n_b};
  float* outputs[2] = {out, out + n_w};

  // Residency (§2d.3): inputs 1..2 are W0 and b0, outputs 0..1 are their updates
  // — the whole parameter set, and the same input-i+1 / output-i correspondence
  // the packed path has, for the same reason (input 0 is x and has no output).
  int rc = use_resident(n_resident)
    ? pjrt_ffi_invoke_f32_resident_v2(sess, fn_name, 1,
        /*res_in=*/1, /*res_out=*/0, (int)n_resident, /*res_gen=*/0,
        4, input_ranks, input_dims_flat, input_data, NULL,
        2, out_totals, outputs)
    : iree_ffi_invoke_f32(sess, fn_name,
        4, input_ranks, input_dims_flat, input_data,
        2, out_totals, outputs);

  free(onehot);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("linear train step failed")));
  }
  return lean_io_result_mk_ok(result);
}

// ---- Data-parallel variant of the verified-renderer packed train step ----
//
// `pjrt_ffi_invoke_f32_dp` is exported only by the XLA shim, so the reference is
// WEAK: the IREE build links fine and simply never reaches this path (the driver
// only calls it when replicas > 1, which the IREE backend never reports).
//
// Shard mask: input 0 is x and the last input is the one-hot, both split across
// replicas; the parameter tensors in between are replicated. `batch` is the
// GLOBAL batch — the shim gives each replica batch/replicas rows.
// ▶ `_dp2` — the `_dp` entry plus `n_shard_tail`. RENAMED rather than extended in place, per §4's
// rule for `pjrt_ffi_invoke_f32_resident_v2`: a stale `.so` paired with a new binary would shift
// every argument, and that is not a link error, it is garbage. A rename makes it a link error.
//
// ⚠⚠ WHY IT EXISTS. `n_shard_tail` is the number of TRAILING param-list inputs that are
// per-example and must therefore be SHARDED like `x`, not replicated like the parameters. Today
// that is exactly the stochastic-depth drop masks (`VerifiedTrain`'s `dropShapes`, `tensor<Bxf32>`
// each), which ride in the parameter blob and so were swept up by "everything between x and the
// labels is replicated". Every replica received replica 0's mask and applied it to its OWN rows —
// `planning/archive/stochastic_depth.md` §5b's predicted defect, sitting in the shim before any DP drop
// render existed to expose it.
//
// ⚠ `PJRT_DP_NO_MASK_SHARD=1` forces the OLD behaviour. It is a deliberate fault-injection knob in
// the `PJRT_FFI_FAULT` tradition, and it exists because a gate nobody has watched go red is not
// evidence: `lake build drop-shard-check` must pass without it and FAIL with it.
LEAN_EXPORT lean_obj_res lean_iree_mlp_train_step_v_dp2(
    b_lean_obj_arg sess_obj, b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg x_ba, b_lean_obj_arg params_ba, b_lean_obj_arg shapes_ba,
    b_lean_obj_arg y_ba, size_t batch, size_t d0, size_t d3, size_t replicas,
    size_t n_resident, size_t n_shard_tail) {
  if (!pjrt_ffi_invoke_f32_dp) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
        "data-parallel train step needs the XLA shim (libpjrt_ffi.so)")));
  }
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int n_inputs = 1 + n_params + 1;
  int32_t* input_ranks = (int32_t*)malloc(n_inputs * sizeof(int32_t));
  int64_t* dims = (int64_t*)malloc((lean_sarray_size(shapes_ba) / 4 + 16) * sizeof(int64_t));
  const float** in_data = (const float**)malloc(n_inputs * sizeof(float*));
  unsigned char* shard = (unsigned char*)calloc(n_inputs, 1);
  int di = 0, sp_idx = 1;

  input_ranks[0] = 2; dims[di++] = (int64_t)batch; dims[di++] = (int64_t)d0;
  in_data[0] = (const float*)lean_sarray_cptr(x_ba);
  shard[0] = 1;                                   // x is sharded

  const float* pf = (const float*)lean_sarray_cptr(params_ba);
  int64_t off = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++]; input_ranks[1 + i] = rank; int64_t sz = 1;
    for (int d = 0; d < rank; d++) { dims[di] = (int64_t)sp[sp_idx++]; sz *= dims[di]; di++; }
    in_data[1 + i] = pf + off; off += sz;         // params replicated (shard stays 0)
  }
  int64_t n_total = off;

  float* onehot = (float*)calloc(batch * d3, sizeof(float));
  if (lean_fill_targets(y_ba, batch, d3, onehot) != 0) {
    free(onehot); free(input_ranks); free(dims); free(in_data); free(shard);
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
        "DP train step: target buffer is neither int32[batch] nor float32[batch*nClasses]")));
  }
  input_ranks[1 + n_params] = 2;
  dims[di++] = (int64_t)batch; dims[di++] = (int64_t)d3;
  in_data[1 + n_params] = onehot;
  shard[1 + n_params] = 1;                        // labels are sharded

  // ▶ The per-example TAIL of the param list — the drop masks. Marked by COUNT, not by index or by
  // shape: an index would be per-net (it depends on nParams/nScalars/nBnStats) and a shape test
  // ("outer dim == batch") would sweep up any parameter that happens to be `batch`-sized. The
  // count comes from the driver, which is the one place that knows how many mask slots it packed.
  {
    const char* off_env = getenv("PJRT_DP_NO_MASK_SHARD");
    int faulted = (off_env && off_env[0] == '1');
    if (n_shard_tail > (size_t)n_params) {
      free(onehot); free(input_ranks); free(dims); free(in_data); free(shard);
      return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
          "DP train step: n_shard_tail exceeds the parameter count")));
    }
    if (!faulted)
      for (size_t k = 0; k < n_shard_tail; k++)
        shard[1 + n_params - 1 - (int)k] = 1;
    else if (n_shard_tail)
      fprintf(stderr, "[pjrt_ffi] ⚠ PJRT_DP_NO_MASK_SHARD=1 — %zu per-example tail input(s) "
                      "REPLICATED instead of sharded (fault injection)\n", n_shard_tail);
  }

  lean_object* result = lean_alloc_sarray(1, (size_t)n_total * 4, (size_t)n_total * 4);
  float* out = (float*)lean_sarray_cptr(result);
  int64_t* out_totals = (int64_t*)malloc(n_params * sizeof(int64_t));
  float** outputs = (float**)malloc(n_params * sizeof(float*));
  sp_idx = 1; off = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++]; int64_t sz = 1;
    for (int d = 0; d < rank; d++) sz *= sp[sp_idx++];
    // ⚠⚠ A SHARDED param-list input that is ALSO an output takes the PER-REPLICA size.
    // Outputs are read from replica 0 only, so it returns its OWN `elems/replicas` rows, not
    // the global buffer it was handed a slice of. Until the drop masks arrived NO sharded
    // input was ever an output (`x` and the labels are inputs only), so this walk could take
    // the declared size for granted. It caught itself rather than reading past the end:
    // `output 740 size mismatch: graph 128 bytes, caller 256` — the same G4 guard that caught
    // the missing BN arity when `shard-check` was generalised to the batch-BN nets.
    if (shard[1 + i] && replicas > 1) sz /= (int64_t)replicas;
    out_totals[i] = sz; outputs[i] = out + off; off += sz;
  }

  // Resident inputs start at 1 (input 0 is x) and the matching outputs start at
  // 0 — the graph returns no counterpart for x, so output i is input i+1.
  int rc = use_resident(n_resident)
    ? pjrt_ffi_invoke_f32_resident_v2(sess, fn_name, (int)replicas,
        /*res_in=*/1, /*res_out=*/0, (int)n_resident, /*res_gen=*/0,
        n_inputs, input_ranks, dims, in_data, shard,
        n_params, out_totals, outputs)
    : pjrt_ffi_invoke_f32_dp(sess, fn_name, (int)replicas,
        n_inputs, input_ranks, dims, in_data, shard,
        n_params, out_totals, outputs);

  free(input_ranks); free(dims); free(in_data); free(shard); free(onehot);
  free(out_totals); free(outputs);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("data-parallel train step failed")));
  }
  return lean_io_result_mk_ok(result);
}

// ---- Verified-renderer MLP train step (StableHLO.mlpTrainStepFaithfulV) ----
// Module signature (x, W0,b0,W1,b1,W2,b2, onehot) -> (W0n,b0n,W1n,b1n,W2n,b2n).
// Inputs: x[batch,d0], the params (packed f32, sliced per `shapes`), onehot
// (built here from int32 labels y[batch], d3 classes). Returns the updated
// params packed in the same layout as `params`.
LEAN_EXPORT lean_obj_res lean_iree_mlp_train_step_v(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg y_ba,
    size_t batch, size_t d0, size_t d3, size_t n_resident) {
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int n_inputs = 1 + n_params + 1;  // x, params..., onehot
  int32_t* input_ranks = (int32_t*)malloc(n_inputs * sizeof(int32_t));
  int64_t* dims = (int64_t*)malloc((lean_sarray_size(shapes_ba) / 4 + 16) * sizeof(int64_t));
  const float** in_data = (const float**)malloc(n_inputs * sizeof(float*));
  int di = 0, sp_idx = 1;

  // input 0: x [batch, d0]
  input_ranks[0] = 2; dims[di++] = (int64_t)batch; dims[di++] = (int64_t)d0;
  in_data[0] = (const float*)lean_sarray_cptr(x_ba);

  // inputs 1..n_params: param tensors sliced from packed `params`
  const float* pf = (const float*)lean_sarray_cptr(params_ba);
  int64_t off = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++]; input_ranks[1 + i] = rank; int64_t sz = 1;
    for (int d = 0; d < rank; d++) { dims[di] = (int64_t)sp[sp_idx++]; sz *= dims[di]; di++; }
    in_data[1 + i] = pf + off; off += sz;
  }
  int64_t n_total = off;

  // last input: target [batch, d3] — int32 hard labels or a float32 distribution (mixup/cutmix)
  float* onehot = (float*)calloc(batch * d3, sizeof(float));
  if (lean_fill_targets(y_ba, batch, d3, onehot) != 0) {
    free(onehot); free(input_ranks); free(dims); free(in_data);
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
        "train step: target buffer is neither int32[batch] nor float32[batch*nClasses]")));
  }
  input_ranks[1 + n_params] = 2; dims[di++] = (int64_t)batch; dims[di++] = (int64_t)d3;
  in_data[1 + n_params] = onehot;

  // outputs: n_params updated tensors, same sizes, packed into one result
  lean_object* result = lean_alloc_sarray(1, (size_t)n_total * 4, (size_t)n_total * 4);
  float* out = (float*)lean_sarray_cptr(result);
  int64_t* out_totals = (int64_t*)malloc(n_params * sizeof(int64_t));
  float** outputs = (float**)malloc(n_params * sizeof(float*));
  sp_idx = 1; off = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++]; int64_t sz = 1;
    for (int d = 0; d < rank; d++) sz *= sp[sp_idx++];
    out_totals[i] = sz; outputs[i] = out + off; off += sz;
  }

  // ---- Optional input dump for hang isolation (env IREE_DUMP_STEP=N) ----
  // Writes the EXACT bytes IREE receives at the N-th invocation of this
  // function, so they can be replayed standalone (FFI-vs-pure-IREE split).
  {
    static int g_call_idx = -1;
    static int g_dump_at = -2;  // -2 unread, -1 disabled
    if (g_dump_at == -2) {
      const char* e = getenv("IREE_DUMP_STEP");
      g_dump_at = e ? atoi(e) : -1;
    }
    g_call_idx++;
    if (g_dump_at >= 0 && g_call_idx == g_dump_at) {
      FILE* fm = fopen("/tmp/dump_meta.txt", "w");
      if (fm) { fprintf(fm, "batch=%zu d0=%zu d3=%zu n_params=%d n_total=%lld\n",
                        batch, d0, d3, n_params, (long long)n_total); fclose(fm); }
      FILE* fx = fopen("/tmp/dump_x.bin", "wb");
      if (fx) { fwrite(in_data[0], sizeof(float), (size_t)batch * d0, fx); fclose(fx); }
      FILE* fp = fopen("/tmp/dump_params.bin", "wb");
      if (fp) { fwrite(pf, sizeof(float), (size_t)n_total, fp); fclose(fp); }
      // The RESOLVED target [batch, d3] f32, not the raw label buffer: since
      // `lean_fill_targets` accepts either int32 labels or a float32 distribution, dumping the
      // input would record two different formats under one filename. This is what the graph got.
      FILE* fy = fopen("/tmp/dump_y.bin", "wb");
      if (fy) { fwrite(onehot, sizeof(float), (size_t)batch * d3, fy); fclose(fy); }
      fprintf(stderr, "[DUMP] wrote step %d inputs (batch=%zu n_total=%lld) to /tmp/dump_*\n",
              g_call_idx, batch, (long long)n_total); fflush(stderr);
    }
  }

  // See the DP peer for the offsets: input 0 is x, so output i is input i+1.
  int rc = use_resident(n_resident)
    ? pjrt_ffi_invoke_f32_resident_v2(sess, fn_name, 1,
        /*res_in=*/1, /*res_out=*/0, (int)n_resident, /*res_gen=*/0,
        n_inputs, input_ranks, dims, in_data, NULL,
        n_params, out_totals, outputs)
    : iree_ffi_invoke_f32(sess, fn_name,
        n_inputs, input_ranks, dims, in_data,
        n_params, out_totals, outputs);

  free(input_ranks); free(dims); free(in_data); free(onehot);
  free(out_totals); free(outputs);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("mlp train step failed")));
  }
  return lean_io_result_mk_ok(result);
}

// ---- Read the authoritative parameter state (§2d.3) ----
// The driver's per-EPOCH `thetamv := pbuf.extract 0 mvBytes`, routed through C so
// that residency is invisible above it. On the copying path (and on IREE, where
// the weak symbol is NULL) this IS that extract, byte for byte. With residency
// live it is the one d2h of the whole blob that still happens, and it happens at
// the frequency it always did — once per epoch, for eval and the checkpoint.
//
// A `rc == 1` from the shim means "this session retains nothing", which is not
// an error: it is the honest answer whenever residency did not engage, and the
// host copy is then authoritative. Only a size disagreement (`rc == 2`) is a
// fault, and it is a loud one — returning a partial parameter state would poison
// a checkpoint silently.
LEAN_EXPORT lean_obj_res lean_iree_read_params(
    b_lean_obj_arg sess_obj, b_lean_obj_arg packed_ba, size_t n_bytes) {
  size_t have = lean_sarray_size(packed_ba);
  if (n_bytes > have) n_bytes = have;
  lean_object* result = lean_alloc_sarray(1, n_bytes, n_bytes);
  uint8_t* dst = lean_sarray_cptr(result);

  if (resident_wanted() && pjrt_ffi_resident_read) {
    iree_ffi_session_t* sess =
        (iree_ffi_session_t*)lean_get_external_data(sess_obj);
    int rc = pjrt_ffi_resident_read(sess, (int64_t)(n_bytes / 4), (float*)dst);
    if (rc == 0) return lean_io_result_mk_ok(result);
    if (rc != 1) {
      lean_dec_ref(result);
      return lean_io_result_mk_error(lean_mk_io_user_error(
          lean_mk_string("resident parameter read-back failed (see stderr)")));
    }
  }
  memcpy(dst, lean_sarray_cptr(packed_ba), n_bytes);
  return lean_io_result_mk_ok(result);
}

// ---- Read the leading parameter tensors only (theta of [theta|m|v]) ----
// `lean_iree_read_params` for a caller that wants theta EVERY step — the DQN runs
// its online-Q forward on the current parameters before each update. Same shape:
// with residency live it is a d2h of theta alone (Adam's m and v stay put); on the
// copying path and on IREE it is the host copy's prefix, byte for byte. A size
// that does not end on a tensor boundary is refused, never rounded.
LEAN_EXPORT lean_obj_res lean_iree_read_params_prefix(
    b_lean_obj_arg sess_obj, b_lean_obj_arg packed_ba, size_t n_bytes) {
  size_t have = lean_sarray_size(packed_ba);
  if (n_bytes > have) n_bytes = have;
  lean_object* result = lean_alloc_sarray(1, n_bytes, n_bytes);
  uint8_t* dst = lean_sarray_cptr(result);

  if (resident_wanted() && pjrt_ffi_resident_read_prefix) {
    iree_ffi_session_t* sess =
        (iree_ffi_session_t*)lean_get_external_data(sess_obj);
    int rc = pjrt_ffi_resident_read_prefix(sess, (int64_t)(n_bytes / 4), (float*)dst);
    if (rc == 0) return lean_io_result_mk_ok(result);
    if (rc != 1) {
      lean_dec_ref(result);
      return lean_io_result_mk_error(lean_mk_io_user_error(
          lean_mk_string("resident parameter prefix read-back failed (see stderr)")));
    }
  }
  memcpy(dst, lean_sarray_cptr(packed_ba), n_bytes);
  return lean_io_result_mk_ok(result);
}
