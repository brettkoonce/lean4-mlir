// F32 ByteArray helpers for Lean FFI.
// All heavy-lift operations (init, read, argmax, data loading) in C to avoid
// millions of Lean-level push calls.

#include <lean/lean.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- Read a float32 at element index from ByteArray ----
LEAN_EXPORT double lean_f32_read(b_lean_obj_arg ba, size_t idx) {
    const float* p = (const float*)lean_sarray_cptr(ba);
    return (double)p[idx];
}

// ---- Fill n float32 values with constant v ----
LEAN_EXPORT lean_obj_res lean_f32_const(size_t n, double v, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    float fv = (float)v;
    for (size_t i = 0; i < n; i++) p[i] = fv;
    return lean_io_result_mk_ok(ba);
}

// ---- He init: n float32 values ~ N(0, scale²) ----
LEAN_EXPORT lean_obj_res lean_f32_he_init(size_t seed, size_t n, double scale, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    uint64_t s = (uint64_t)seed + 1;
    float fscale = (float)scale;
    for (size_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (int k = 0; k < 3; k++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            acc += (float)s / (float)UINT64_MAX - 0.5f;
        }
        p[i] = acc * 2.0f * fscale;
    }
    return lean_io_result_mk_ok(ba);
}

// ---- Argmax over 10 float32 values at element offset ----
LEAN_EXPORT size_t lean_f32_argmax10(b_lean_obj_arg ba, size_t off) {
    const float* p = (const float*)lean_sarray_cptr(ba);
    size_t best = 0;
    float bestv = p[off];
    for (size_t i = 1; i < 10; i++) {
        if (p[off + i] > bestv) { best = i; bestv = p[off + i]; }
    }
    return best;
}

// ---- Convert a batch of CIFAR-10 raw records to f32 ByteArray ----
// Each record is 3073 bytes (1 label + 3072 pixels). Normalizes to [0,1].
LEAN_EXPORT lean_obj_res lean_f32_cifar_batch(
    b_lean_obj_arg raw_ba, size_t start, size_t count, lean_obj_arg w) {
    (void)w;
    const uint8_t* raw = lean_sarray_cptr(raw_ba);
    size_t npixels = count * 3072;
    size_t nbytes = npixels * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    for (size_t i = 0; i < count; i++) {
        size_t rec_off = (start + i) * 3073;
        for (size_t j = 0; j < 3072; j++) {
            p[i * 3072 + j] = (float)raw[rec_off + 1 + j] / 255.0f;
        }
    }
    return lean_io_result_mk_ok(ba);
}

// ---- Load MNIST IDX images → f32 ByteArray (normalized to [0,1]) ----
static uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

LEAN_EXPORT lean_obj_res lean_f32_load_idx_images(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("cannot open image file")));
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* raw = (uint8_t*)malloc(fsize);
    if (fread(raw, 1, fsize, f) != (size_t)fsize) { free(raw); fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("short read on IDX image file"))); }
    fclose(f);

    uint32_t magic = read_be32(raw);
    if (magic != 2051) { free(raw);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad IDX image magic"))); }
    uint32_t n = read_be32(raw + 4);
    uint32_t rows = read_be32(raw + 8);
    uint32_t cols = read_be32(raw + 12);
    size_t total = (size_t)n * rows * cols;

    size_t nbytes = total * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    for (size_t i = 0; i < total; i++)
        p[i] = (float)raw[16 + i] / 255.0f;
    free(raw);

    // Return (ByteArray, USize) as a pair
    // Prod ByteArray Nat: 2 boxed object fields
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, ba);
    lean_ctor_set(pair, 1, lean_usize_to_nat((size_t)n));
    return lean_io_result_mk_ok(pair);
}

// ---- Load MNIST IDX labels → int32 LE ByteArray ----
LEAN_EXPORT lean_obj_res lean_f32_load_idx_labels(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("cannot open label file")));
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* raw = (uint8_t*)malloc(fsize);
    if (fread(raw, 1, fsize, f) != (size_t)fsize) { free(raw); fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("short read on IDX label file"))); }
    fclose(f);

    uint32_t magic = read_be32(raw);
    if (magic != 2049) { free(raw);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad IDX label magic"))); }
    uint32_t n = read_be32(raw + 4);

    size_t nbytes = (size_t)n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    uint8_t* out = lean_sarray_cptr(ba);
    for (uint32_t i = 0; i < n; i++) {
        out[i * 4]     = raw[8 + i];
        out[i * 4 + 1] = 0;
        out[i * 4 + 2] = 0;
        out[i * 4 + 3] = 0;
    }
    free(raw);

    // Prod ByteArray Nat: 2 boxed object fields
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, ba);
    lean_ctor_set(pair, 1, lean_usize_to_nat((size_t)n));
    return lean_io_result_mk_ok(pair);
}

// ---- Imagenette binary -> f32 ByteArray (ImageNet mean/std normalized) ----
// Binary: 4-byte count (LE u32), per-sample: 1 byte label + 224*224*3 bytes (CHW, uint8)
// Returns (images ByteArray, labels ByteArray, count Nat)
// Internal loader parameterized by image size
static lean_obj_res load_imagenette_sized(const char* path, size_t img_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open imagenette file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f); return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad header"))); }
    uint32_t count = file_count;
    const size_t pix = 3 * img_size * img_size;
    size_t img_bytes = (size_t)count * pix * 4;
    size_t lbl_bytes = (size_t)count * 4;
    lean_object* img_ba = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* lbl_ba = lean_alloc_sarray(1, lbl_bytes, lbl_bytes);
    float* img = (float*)lean_sarray_cptr(img_ba);
    uint8_t* lbl = lean_sarray_cptr(lbl_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(1 + pix);
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, 1 + pix, f) != 1 + pix) { free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("short read"))); }
        lbl[i*4]=buf[0]; lbl[i*4+1]=0; lbl[i*4+2]=0; lbl[i*4+3]=0;
        float* dst = img + (size_t)i * pix;
        size_t hw = img_size * img_size;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[1+ch*hw+j]/255.0f - m) * s;
        }
    }
    free(buf); fclose(f);

    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, lbl_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

LEAN_EXPORT lean_obj_res lean_f32_load_imagenette(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    return load_imagenette_sized(lean_string_cstr(path_obj), 224);
}

LEAN_EXPORT lean_obj_res lean_f32_load_imagenette_sized(b_lean_obj_arg path_obj, size_t img_size, lean_obj_arg w) {
    (void)w;
    return load_imagenette_sized(lean_string_cstr(path_obj), img_size);
}

// ---- Pets (Oxford-IIIT) loader ----
// Binary format per record:
//   image: 3 * 224 * 224 bytes (channel-first RGB, uint8)
//   mask:  224 * 224     bytes (per-pixel class 0/1/2)
// Total per record: 200,704 bytes.
//
// Returns (image_f32_normalized, mask_uint8, count) as a 3-tuple.
// Image is normalized with ImageNet mean/std for transfer-learning consistency
// with the rest of the project; the mask is left as raw uint8 (one byte per
// pixel) so downstream code can treat it as integer per-pixel class labels.
LEAN_EXPORT lean_obj_res lean_f32_load_pets(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open pets file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad header"))); }
    uint32_t count = file_count;
    const size_t img_size = 224;
    const size_t pix = 3 * img_size * img_size;          // 150,528
    const size_t mask_pix = img_size * img_size;         // 50,176
    size_t img_bytes  = (size_t)count * pix * 4;         // f32 image buffer
    size_t mask_bytes = (size_t)count * mask_pix;        // uint8 mask buffer
    lean_object* img_ba  = lean_alloc_sarray(1, img_bytes,  img_bytes);
    lean_object* mask_ba = lean_alloc_sarray(1, mask_bytes, mask_bytes);
    float*   img  = (float*)lean_sarray_cptr(img_ba);
    uint8_t* mask = lean_sarray_cptr(mask_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(pix + mask_pix);
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, pix + mask_pix, f) != pix + mask_pix) { free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("short read"))); }
        // Normalize image into f32 buffer
        float* dst = img + (size_t)i * pix;
        size_t hw = img_size * img_size;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[ch*hw+j]/255.0f - m) * s;
        }
        // Mask is already uint8 0/1/2; copy directly
        memcpy(mask + (size_t)i * mask_pix, buf + pix, mask_pix);
    }
    free(buf); fclose(f);

    lean_object* inner_pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner_pair, 0, mask_ba);
    lean_ctor_set(inner_pair, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner_pair);
    return lean_io_result_mk_ok(outer);
}

// ---- BraTS (MSD Task01_BrainTumour) loader ----
// Binary format per record (written by preprocess_brats.py):
//   image: 4 * S * S bytes (channel-first [modality][y][x], uint8)
//   mask:  S * S     bytes (per-pixel class 0..3)
// At the default S=240: 288,000 bytes/record.
//
// Returns (image_f32_dequantized, mask_uint8, count) as a 3-tuple — the same
// shape as lean_f32_load_pets, so the segmentation train path is unchanged.
//
// Unlike the pets loader, this does NOT apply ImageNet mean/std. MRI is not
// RGB and has no such statistics. preprocess_brats.py z-scores each modality
// over that volume's brain (nonzero) voxels and quantizes the result to uint8
// over a +/-BRATS_CLIP_SIGMA window; here we invert exactly that, so the model
// sees per-volume z-scored intensities. The inverse must stay in lockstep with
// quantize_u8() in preprocess_brats.py.
#define BRATS_CLIP_SIGMA 5.0f
#define BRATS_CHANNELS 4

LEAN_EXPORT lean_obj_res lean_f32_load_brats(b_lean_obj_arg path_obj, size_t img_size, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open brats file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad header"))); }
    uint32_t count = file_count;
    const size_t hw = img_size * img_size;
    const size_t pix = BRATS_CHANNELS * hw;
    const size_t mask_pix = hw;
    size_t img_bytes  = (size_t)count * pix * 4;         // f32 image buffer
    size_t mask_bytes = (size_t)count * mask_pix;        // uint8 mask buffer
    lean_object* img_ba  = lean_alloc_sarray(1, img_bytes,  img_bytes);
    lean_object* mask_ba = lean_alloc_sarray(1, mask_bytes, mask_bytes);
    float*   img  = (float*)lean_sarray_cptr(img_ba);
    uint8_t* mask = lean_sarray_cptr(mask_ba);
    // Inverse of preprocess_brats.quantize_u8: z = (b - 128) * (CLIP/127).
    // Zero is anchored at 128 there, so background (exact zero in a
    // skull-stripped volume) round-trips to exact zero here.
    const float dequant = BRATS_CLIP_SIGMA / 127.0f;
    uint8_t* buf = (uint8_t*)malloc(pix + mask_pix);
    if (!buf) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("brats: malloc failed"))); }
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, pix + mask_pix, f) != pix + mask_pix) { free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("short read"))); }
        float* dst = img + (size_t)i * pix;
        for (size_t j = 0; j < pix; j++)
            dst[j] = ((float)buf[j] - 128.0f) * dequant;
        // Mask is already uint8 0..3; copy directly.
        memcpy(mask + (size_t)i * mask_pix, buf + pix, mask_pix);
    }
    free(buf); fclose(f);

    lean_object* inner_pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner_pair, 0, mask_ba);
    lean_ctor_set(inner_pair, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner_pair);
    return lean_io_result_mk_ok(outer);
}

// ---- VOC 2007 YOLOv1 binary loader ----
// Binary format (written by preprocess_voc.py — Phase 3b layout, 157,728
// bytes/record):
//   Header: 4-byte LE uint32 count
//   Per record (157,728 bytes total):
//     image      : 3 * 224 * 224     bytes uint8   (channel-first RGB)
//     target     : 30 * 7 * 7 * 4    bytes float32 (NCHW)
//     mask       :      7 * 7 * 4    bytes float32 (per-cell objectness)
//     numBoxes   :              4    bytes int32 LE
//     raw_boxes  : 56 * 20           bytes (each: 4-byte i32 class +
//                                    4 × 4 byte f32 xmin/ymin/xmax/ymax
//                                    normalized [0,1])
//
// Returns (image_f32_normalized, ylabels_concat, count) where
// ylabels_concat lays out target ++ mask ++ numBoxes ++ raw_boxes per
// record (7,200 bytes/record). The Lean dispatcher splits this at
// training time. Image is ImageNet-normalized as in load_imagenette/
// load_pets. See planning/yolo_demo_v2.md Phase 1 + planning/yolo_demo_v3.md
// Phase 2-3.
LEAN_EXPORT lean_obj_res lean_f32_load_voc(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open voc file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad voc header"))); }
    uint32_t count = file_count;
    const size_t img_size = 224;
    const size_t pix = 3 * img_size * img_size;          // 150,528
    const size_t target_bytes_per = 30 * 7 * 7 * 4;      // 5,880
    const size_t mask_bytes_per   = 7 * 7 * 4;           // 196
    const size_t nbox_bytes_per   = 4;
    const size_t boxes_bytes_per  = 56 * 20;             // 1,120
    const size_t aux_bytes_per    = target_bytes_per + mask_bytes_per
                                    + nbox_bytes_per + boxes_bytes_per; // 7,200
    size_t img_bytes  = (size_t)count * pix * 4;         // f32 image buffer
    size_t aux_bytes  = (size_t)count * aux_bytes_per;
    lean_object* img_ba = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* aux_ba = lean_alloc_sarray(1, aux_bytes, aux_bytes);
    float*   img = (float*)lean_sarray_cptr(img_ba);
    uint8_t* aux = lean_sarray_cptr(aux_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(pix + aux_bytes_per);
    if (!buf) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("voc loader: malloc failed"))); }
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, pix + aux_bytes_per, f) != pix + aux_bytes_per) {
            free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("voc loader: short read")));
        }
        // Normalize image into f32 buffer.
        float* dst = img + (size_t)i * pix;
        size_t hw = img_size * img_size;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[ch*hw+j]/255.0f - m) * s;
        }
        // Copy target ++ mask ++ numBoxes ++ raw_boxes verbatim.
        memcpy(aux + (size_t)i * aux_bytes_per, buf + pix, aux_bytes_per);
    }
    free(buf); fclose(f);

    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, aux_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

// Dimension-parameterized YOLOv1 detection-bin loader. Identical to
// lean_f32_load_voc but takes (imgSize, gridH, gridW) so the same record
// format can be used at higher resolution / finer grid (e.g. VisDrone at
// 448 input / 14x14 grid) without a separate hardcoded copy. perCell (30 =
// 2 boxes*5 + 20 classes) and MAX_BBOXES (56) are fixed by the layout.
// On-disk record: image 3*imgSize*imgSize uint8, target 30*gridH*gridW f32,
// mask gridH*gridW f32, numBoxes i32, raw_boxes 56*20. Returns
// (image_f32_normalized, target++mask++numBoxes++raw_boxes concat, count).
LEAN_EXPORT lean_obj_res lean_f32_load_voc_dims(
        b_lean_obj_arg path_obj, size_t imgSize, size_t gridH, size_t gridW,
        lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open voc file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad voc header"))); }
    uint32_t count = file_count;
    const size_t pix = 3 * imgSize * imgSize;
    const size_t target_bytes_per = 30 * gridH * gridW * 4;
    const size_t mask_bytes_per   = gridH * gridW * 4;
    const size_t nbox_bytes_per   = 4;
    const size_t boxes_bytes_per  = 56 * 20;
    const size_t aux_bytes_per    = target_bytes_per + mask_bytes_per
                                    + nbox_bytes_per + boxes_bytes_per;
    size_t img_bytes  = (size_t)count * pix * 4;
    size_t aux_bytes  = (size_t)count * aux_bytes_per;
    lean_object* img_ba = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* aux_ba = lean_alloc_sarray(1, aux_bytes, aux_bytes);
    float*   img = (float*)lean_sarray_cptr(img_ba);
    uint8_t* aux = lean_sarray_cptr(aux_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(pix + aux_bytes_per);
    if (!buf) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("voc loader: malloc failed"))); }
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, pix + aux_bytes_per, f) != pix + aux_bytes_per) {
            free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("voc loader: short read")));
        }
        float* dst = img + (size_t)i * pix;
        size_t hw = imgSize * imgSize;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[ch*hw+j]/255.0f - m) * s;
        }
        memcpy(aux + (size_t)i * aux_bytes_per, buf + pix, aux_bytes_per);
    }
    free(buf); fclose(f);

    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, aux_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

// Anchor-format detection loader (brick #2). On-disk record:
//   image 3*imgSize^2 uint8, target A*15*gH*gW f32, mask A*gH*gW f32,
//   numBoxes i32, raw_boxes 56*20 (as preprocess_visdrone.py --anchors writes).
// Returns (image_f32_normalized, target_only_concat, count): only the target is
// returned (A*15*gH*gW f32/record) — the anchor loss derives its per-anchor mask
// from the target's objectness channels, so mask/numBoxes/raw_boxes are skipped.
LEAN_EXPORT lean_obj_res lean_f32_load_voc_anchor(
        b_lean_obj_arg path_obj, size_t imgSize, size_t gridH, size_t gridW,
        size_t numAnchors, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open anchor file")));
    uint32_t count;
    if (fread(&count, 4, 1, f) != 1) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad anchor header"))); }
    const size_t pix = 3 * imgSize * imgSize;
    const size_t tgt_bytes = numAnchors * 15 * gridH * gridW * 4;
    const size_t msk_bytes = numAnchors * gridH * gridW * 4;
    const size_t skip_bytes = msk_bytes + 4 + 56 * 20;      // mask + numBoxes + raw_boxes
    const size_t rec_bytes = pix + tgt_bytes + skip_bytes;
    size_t img_bytes = (size_t)count * pix * 4;
    size_t tgt_total = (size_t)count * tgt_bytes;
    lean_object* img_ba = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* tgt_ba = lean_alloc_sarray(1, tgt_total, tgt_total);
    float* img = (float*)lean_sarray_cptr(img_ba);
    uint8_t* tgt = lean_sarray_cptr(tgt_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(rec_bytes);
    if (!buf) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("anchor loader: malloc failed"))); }
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, rec_bytes, f) != rec_bytes) {
            free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("anchor loader: short read")));
        }
        float* dst = img + (size_t)i * pix;
        size_t hw = imgSize * imgSize;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[ch*hw+j]/255.0f - m) * s;
        }
        memcpy(tgt + (size_t)i * tgt_bytes, buf + pix, tgt_bytes);   // target only
    }
    free(buf); fclose(f);

    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, tgt_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

// FPN multi-scale detection loader (brick #3). On-disk record:
//   image 3*imgSize^2 uint8, target ntot f32 (flat [P3|P4|P5] block,
//   ntot = sum_s numAnchors_s*15*g_s^2, as preprocess_visdrone.py --fpn writes).
// Returns (image_f32_normalized, target_only_concat, count) — image + flat target
// only (no mask/boxes on disk; the loss derives masks from the target's obj
// channels, and eval GT comes from the single-box val.bin geometry).
LEAN_EXPORT lean_obj_res lean_f32_load_voc_fpn(
        b_lean_obj_arg path_obj, size_t imgSize, size_t ntot, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open fpn file")));
    uint32_t count;
    if (fread(&count, 4, 1, f) != 1) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad fpn header"))); }
    const size_t pix = 3 * imgSize * imgSize;
    const size_t tgt_bytes = ntot * 4;
    const size_t rec_bytes = pix + tgt_bytes;
    size_t img_bytes = (size_t)count * pix * 4;
    size_t tgt_total = (size_t)count * tgt_bytes;
    lean_object* img_ba = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* tgt_ba = lean_alloc_sarray(1, tgt_total, tgt_total);
    float* img = (float*)lean_sarray_cptr(img_ba);
    uint8_t* tgt = lean_sarray_cptr(tgt_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(rec_bytes);
    if (!buf) { fclose(f);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("fpn loader: malloc failed"))); }
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, rec_bytes, f) != rec_bytes) {
            free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("fpn loader: short read")));
        }
        float* dst = img + (size_t)i * pix;
        size_t hw = imgSize * imgSize;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[ch*hw+j]/255.0f - m) * s;
        }
        memcpy(tgt + (size_t)i * tgt_bytes, buf + pix, tgt_bytes);   // flat target
    }
    free(buf); fclose(f);

    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, tgt_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

// Re-encode YOLOv1 target+mask tensors from a list of bboxes. Matches
// the encode_targets() in preprocess_voc.py — used by lean_f32_yolo_augment
// after geometric transforms (hflip, random crop) shift the boxes around.
// Boxes layout: per box (cid, xmin, ymin, xmax, ymax) as 5 contiguous floats
// (cid is stored as float and cast to int on use, matching the format the
// transform_box helper produces). xmin/ymin/xmax/ymax are normalized [0,1].
static void yolo_encode_target_mask(
    const float* boxes, int n_boxes,
    float* target, float* mask,
    int perCell, int gridH, int gridW, int numClasses) {
    memset(target, 0, (size_t)perCell * gridH * gridW * sizeof(float));
    memset(mask,   0, (size_t)gridH * gridW * sizeof(float));
    for (int i = 0; i < n_boxes; i++) {
        int cid = (int)boxes[i * 5 + 0];
        float xmin = boxes[i * 5 + 1];
        float ymin = boxes[i * 5 + 2];
        float xmax = boxes[i * 5 + 3];
        float ymax = boxes[i * 5 + 4];
        float cx = 0.5f * (xmin + xmax);
        float cy = 0.5f * (ymin + ymax);
        float w_rel = xmax - xmin;
        float h_rel = ymax - ymin;
        int cell_j = (int)(cx * (float)gridW);
        int cell_i = (int)(cy * (float)gridH);
        if (cell_j >= gridW) cell_j = gridW - 1;
        if (cell_i >= gridH) cell_i = gridH - 1;
        if (cell_j < 0) cell_j = 0;
        if (cell_i < 0) cell_i = 0;
        float cx_cell = cx * (float)gridW - (float)cell_j;
        float cy_cell = cy * (float)gridH - (float)cell_i;
        size_t cell_off = (size_t)cell_i * (size_t)gridW + (size_t)cell_j;
        size_t plane   = (size_t)gridH * (size_t)gridW;
        // Clear any class one-hot at this cell from a prior (overwritten) box.
        for (int c = 0; c < numClasses; c++) {
            target[(size_t)(10 + c) * plane + cell_off] = 0.0f;
        }
        target[(size_t)0 * plane + cell_off] = cx_cell;
        target[(size_t)1 * plane + cell_off] = cy_cell;
        target[(size_t)2 * plane + cell_off] = w_rel;
        target[(size_t)3 * plane + cell_off] = h_rel;
        target[(size_t)4 * plane + cell_off] = 1.0f;  // box 0 conf
        if (cid >= 0 && cid < numClasses) {
            target[(size_t)(10 + cid) * plane + cell_off] = 1.0f;
        }
        mask[cell_off] = 1.0f;
    }
}

// Bilinear resize from a sub-region of the source image to the full
// destination. `src_off_y/x` + `crop_h/w` define the sub-region (in source
// pixel coords); `hflip` reverses the output W axis (sampled by mirroring
// the output x before the source-coord computation).
static void yolo_bilinear_resize_chw(
    const float* src, int src_h, int src_w, int channels,
    float* dst, int dst_h, int dst_w,
    int src_off_y, int src_off_x, int crop_h, int crop_w,
    int hflip) {
    if (crop_h < 1) crop_h = 1;
    if (crop_w < 1) crop_w = 1;
    size_t src_plane = (size_t)src_h * (size_t)src_w;
    size_t dst_plane = (size_t)dst_h * (size_t)dst_w;
    for (int oy = 0; oy < dst_h; oy++) {
        float fy = ((float)oy + 0.5f) * (float)crop_h / (float)dst_h - 0.5f;
        if (fy < 0.0f) fy = 0.0f;
        if (fy > (float)(crop_h - 1)) fy = (float)(crop_h - 1);
        int iy0 = (int)fy;
        int iy1 = iy0 + 1; if (iy1 >= crop_h) iy1 = crop_h - 1;
        float wy = fy - (float)iy0;
        int sy0 = src_off_y + iy0;
        int sy1 = src_off_y + iy1;
        for (int ox = 0; ox < dst_w; ox++) {
            int ox_src = hflip ? (dst_w - 1 - ox) : ox;
            float fx = ((float)ox_src + 0.5f) * (float)crop_w / (float)dst_w - 0.5f;
            if (fx < 0.0f) fx = 0.0f;
            if (fx > (float)(crop_w - 1)) fx = (float)(crop_w - 1);
            int ix0 = (int)fx;
            int ix1 = ix0 + 1; if (ix1 >= crop_w) ix1 = crop_w - 1;
            float wx = fx - (float)ix0;
            int sx0 = src_off_x + ix0;
            int sx1 = src_off_x + ix1;
            for (int ch = 0; ch < channels; ch++) {
                float p00 = src[(size_t)ch * src_plane + (size_t)sy0 * src_w + sx0];
                float p01 = src[(size_t)ch * src_plane + (size_t)sy0 * src_w + sx1];
                float p10 = src[(size_t)ch * src_plane + (size_t)sy1 * src_w + sx0];
                float p11 = src[(size_t)ch * src_plane + (size_t)sy1 * src_w + sx1];
                float interp = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p01)
                             + wy        * ((1.0f - wx) * p10 + wx * p11);
                dst[(size_t)ch * dst_plane + (size_t)oy * dst_w + ox] = interp;
            }
        }
    }
}

// Transform a single bbox through (random crop, then hflip). Inputs are in
// [0, 1] of the original image; outputs are in [0, 1] of the cropped+resized
// image. Returns 1 if the box survives (≥50% of original area inside crop
// AND non-empty), 0 if dropped.
static int yolo_transform_box(
    const float* src_box, float* dst_box, int hflip,
    float crop_off_x_rel, float crop_off_y_rel,
    float crop_w_rel, float crop_h_rel) {
    int cid    = (int)src_box[0];
    float xmin = src_box[1];
    float ymin = src_box[2];
    float xmax = src_box[3];
    float ymax = src_box[4];
    float new_xmin = (xmin - crop_off_x_rel) / crop_w_rel;
    float new_xmax = (xmax - crop_off_x_rel) / crop_w_rel;
    float new_ymin = (ymin - crop_off_y_rel) / crop_h_rel;
    float new_ymax = (ymax - crop_off_y_rel) / crop_h_rel;
    if (new_xmin < 0.0f) new_xmin = 0.0f;
    if (new_xmax > 1.0f) new_xmax = 1.0f;
    if (new_ymin < 0.0f) new_ymin = 0.0f;
    if (new_ymax > 1.0f) new_ymax = 1.0f;
    if (new_xmax <= new_xmin || new_ymax <= new_ymin) return 0;
    float old_w = xmax - xmin, old_h = ymax - ymin;
    float new_w = new_xmax - new_xmin, new_h = new_ymax - new_ymin;
    if (old_w * old_h > 0.0f && (new_w * new_h) / (old_w * old_h) < 0.5f) return 0;
    if (hflip) {
        float tmp = new_xmin;
        new_xmin = 1.0f - new_xmax;
        new_xmax = 1.0f - tmp;
    }
    dst_box[0] = (float)cid;
    dst_box[1] = new_xmin;
    dst_box[2] = new_ymin;
    dst_box[3] = new_xmax;
    dst_box[4] = new_ymax;
    return 1;
}

// Unified bbox-aware augmentation for YOLOv1: per-image hflip + random crop,
// with the target+mask re-encoded from the transformed bboxes (so the
// geometric correspondence is exact). Replaces the simpler lean_f32_yolo_hflip
// from Phase 3a now that the data file carries raw bboxes (Phase 3b).
//
//   `img_ba`   : f32 image batch [B, C, H, W]
//   `box_ba`   : YOLOv1 label block per record (target 5880 + mask 196 +
//                numBoxes (i32, 4) + raw_boxes (56 × 20)) = 7200 bytes/record.
//                Only the numBoxes + raw_boxes tail is read by this kernel;
//                the leading target+mask block is recomputed.
//   `batch`    : B
//   Geometry args: channels, imgH, imgW, gridH, gridW, perCell, numClasses
//   Probabilities: hflip_prob (0-1), crop_prob (0-1) — applied per image.
//   Crop intensity: crop_min_scale (e.g. 0.8) — crop_size ∈ [min_scale, 1] × img.
//   `seed`     : xorshift seed
//
// Returns (new_image, new_target, new_mask) triple of fresh ByteArrays.
// See planning/yolo_demo_v3.md Phase 3.
LEAN_EXPORT lean_obj_res lean_f32_yolo_augment(
    b_lean_obj_arg img_ba, b_lean_obj_arg box_ba,
    size_t batch, size_t channels, size_t imgH, size_t imgW,
    size_t gridH, size_t gridW, size_t perCell, size_t numClasses,
    double hflip_prob, double crop_prob, double crop_min_scale,
    size_t seed, lean_obj_arg w) {
    (void)w;
    const size_t img_floats_per   = channels * imgH * imgW;
    const size_t tgt_floats_per   = perCell * gridH * gridW;
    const size_t msk_floats_per   = gridH * gridW;
    const size_t img_bytes_per    = img_floats_per * 4;
    const size_t tgt_bytes_per    = tgt_floats_per * 4;
    const size_t msk_bytes_per    = msk_floats_per * 4;
    const size_t box_rec_bytes    = tgt_bytes_per + msk_bytes_per + 4 + 56 * 20;  // 7200
    const size_t box_tail_off     = tgt_bytes_per + msk_bytes_per;  // skip to numBoxes
    lean_object* out_img_ba = lean_alloc_sarray(1, batch * img_bytes_per, batch * img_bytes_per);
    lean_object* out_tgt_ba = lean_alloc_sarray(1, batch * tgt_bytes_per, batch * tgt_bytes_per);
    lean_object* out_msk_ba = lean_alloc_sarray(1, batch * msk_bytes_per, batch * msk_bytes_per);
    const float*   src_img = (const float*)lean_sarray_cptr(img_ba);
    const uint8_t* src_box = (const uint8_t*)lean_sarray_cptr(box_ba);
    float* dst_img = (float*)lean_sarray_cptr(out_img_ba);
    float* dst_tgt = (float*)lean_sarray_cptr(out_tgt_ba);
    float* dst_msk = (float*)lean_sarray_cptr(out_msk_ba);

    uint64_t rng = seed ? (uint64_t)seed : 0x9E3779B97F4A7C15ULL;
    float transformed[56 * 5];  // scratch for transformed bboxes per image
    for (size_t b = 0; b < batch; b++) {
        // Per-image RNG draws.
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double r_hflip = (double)((rng >> 32) & 0xFFFFFF) / (double)0xFFFFFF;
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double r_crop  = (double)((rng >> 32) & 0xFFFFFF) / (double)0xFFFFFF;
        int hflip = (r_hflip < hflip_prob) ? 1 : 0;
        int crop  = (r_crop  < crop_prob)  ? 1 : 0;

        int crop_w = (int)imgW, crop_h = (int)imgH;
        int off_x = 0, off_y = 0;
        if (crop) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double rs = (double)((rng >> 32) & 0xFFFFFF) / (double)0xFFFFFF;
            double scale = crop_min_scale + (1.0 - crop_min_scale) * rs;
            crop_w = (int)(scale * (double)imgW);
            crop_h = crop_w;
            if (crop_w > (int)imgW) crop_w = (int)imgW;
            if (crop_h > (int)imgH) crop_h = (int)imgH;
            if (crop_w < 1) crop_w = 1;
            if (crop_h < 1) crop_h = 1;
            int max_off_x = (int)imgW - crop_w;
            int max_off_y = (int)imgH - crop_h;
            if (max_off_x > 0) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                off_x = (int)(((rng >> 32) & 0xFFFFFF) % (uint64_t)(max_off_x + 1));
            }
            if (max_off_y > 0) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                off_y = (int)(((rng >> 32) & 0xFFFFFF) % (uint64_t)(max_off_y + 1));
            }
        }

        // Image: bilinear resize the crop region back to full size, with optional hflip.
        yolo_bilinear_resize_chw(
            src_img + b * img_floats_per,
            (int)imgH, (int)imgW, (int)channels,
            dst_img + b * img_floats_per,
            (int)imgH, (int)imgW,
            off_y, off_x, crop_h, crop_w, hflip);

        // Bboxes: read numBoxes + raw_boxes from the per-record label block.
        const uint8_t* rec = src_box + b * box_rec_bytes;
        int32_t n_boxes;
        memcpy(&n_boxes, rec + box_tail_off, 4);
        if (n_boxes > 56) n_boxes = 56;
        const uint8_t* boxes_arr = rec + box_tail_off + 4;
        float crop_off_x_rel = (float)off_x / (float)imgW;
        float crop_off_y_rel = (float)off_y / (float)imgH;
        float crop_w_rel = (float)crop_w / (float)imgW;
        float crop_h_rel = (float)crop_h / (float)imgH;
        int n_kept = 0;
        for (int i = 0; i < n_boxes; i++) {
            const uint8_t* bp = boxes_arr + i * 20;
            int32_t cid;
            memcpy(&cid, bp, 4);
            float coords[4];
            memcpy(coords, bp + 4, 16);
            float src_box_arr[5] = {(float)cid, coords[0], coords[1], coords[2], coords[3]};
            if (yolo_transform_box(src_box_arr, transformed + n_kept * 5, hflip,
                                   crop_off_x_rel, crop_off_y_rel,
                                   crop_w_rel, crop_h_rel)) {
                n_kept++;
            }
        }

        // Re-encode target+mask from surviving bboxes.
        yolo_encode_target_mask(
            transformed, n_kept,
            dst_tgt + b * tgt_floats_per,
            dst_msk + b * msk_floats_per,
            (int)perCell, (int)gridH, (int)gridW, (int)numClasses);
    }

    lean_object* inner_pair = lean_alloc_ctor(0, 2, 0);  // (target, mask)
    lean_ctor_set(inner_pair, 0, out_tgt_ba);
    lean_ctor_set(inner_pair, 1, out_msk_ba);
    lean_object* outer = lean_alloc_ctor(0, 2, 0);       // (image, (target, mask))
    lean_ctor_set(outer, 0, out_img_ba);
    lean_ctor_set(outer, 1, inner_pair);
    return lean_io_result_mk_ok(outer);
}

// Bbox-aware horizontal flip for YOLOv1 batches. Per-image coin flip
// (xorshift64 from `seed`); when flipped:
//   * image  [B, C, H, W]: reverse the W axis for every (c, h) row
//   * target [B, perCell, gH, gW]: reverse the gW axis for every
//     (perCell, gH) row, then on cells where mask=1 replace the x_cell
//     channel (channel 0 — box 0's x) with 1 - x_cell because the cell
//     itself is mirrored. Other channels (y, w, h, conf, class one-hot,
//     box-1 slots) are hflip-symmetric and stay put.
//   * mask   [B, gH, gW]: reverse the gW axis for every gH row
//
// Returns (images, target, mask) as a 3-tuple of fresh ByteArrays.
// Inputs are not modified. See planning/yolo_demo_v3.md Phase 3.
//
// LEGACY: superseded by lean_f32_yolo_augment (Phase 3b) which operates
// on raw bboxes and re-encodes target+mask from scratch. Kept for the
// hflip-only smoke path; once the v3 trainer commits fully to the
// augment kernel this can be removed.
LEAN_EXPORT lean_obj_res lean_f32_yolo_hflip(
    b_lean_obj_arg img_ba, b_lean_obj_arg tgt_ba, b_lean_obj_arg msk_ba,
    size_t batch, size_t channels, size_t imgH, size_t imgW,
    size_t gridH, size_t gridW, size_t perCell, size_t seed,
    lean_obj_arg w) {
    (void)w;
    const size_t img_floats_per   = channels * imgH * imgW;
    const size_t tgt_floats_per   = perCell * gridH * gridW;
    const size_t msk_floats_per   = gridH * gridW;
    const size_t img_bytes_per    = img_floats_per * 4;
    const size_t tgt_bytes_per    = tgt_floats_per * 4;
    const size_t msk_bytes_per    = msk_floats_per * 4;
    const size_t total_img_bytes  = batch * img_bytes_per;
    const size_t total_tgt_bytes  = batch * tgt_bytes_per;
    const size_t total_msk_bytes  = batch * msk_bytes_per;

    lean_object* out_img_ba = lean_alloc_sarray(1, total_img_bytes, total_img_bytes);
    lean_object* out_tgt_ba = lean_alloc_sarray(1, total_tgt_bytes, total_tgt_bytes);
    lean_object* out_msk_ba = lean_alloc_sarray(1, total_msk_bytes, total_msk_bytes);
    const float* src_img = (const float*)lean_sarray_cptr(img_ba);
    const float* src_tgt = (const float*)lean_sarray_cptr(tgt_ba);
    const float* src_msk = (const float*)lean_sarray_cptr(msk_ba);
    float* dst_img = (float*)lean_sarray_cptr(out_img_ba);
    float* dst_tgt = (float*)lean_sarray_cptr(out_tgt_ba);
    float* dst_msk = (float*)lean_sarray_cptr(out_msk_ba);

    uint64_t rng = seed ? (uint64_t)seed : 0x9E3779B97F4A7C15ULL;
    for (size_t b = 0; b < batch; b++) {
        // xorshift64 next bit.
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        int flip = (int)(rng & 1);

        const float* sI = src_img + b * img_floats_per;
        const float* sT = src_tgt + b * tgt_floats_per;
        const float* sM = src_msk + b * msk_floats_per;
        float* dI = dst_img + b * img_floats_per;
        float* dT = dst_tgt + b * tgt_floats_per;
        float* dM = dst_msk + b * msk_floats_per;

        if (!flip) {
            memcpy(dI, sI, img_bytes_per);
            memcpy(dT, sT, tgt_bytes_per);
            memcpy(dM, sM, msk_bytes_per);
            continue;
        }

        // Image: reverse W per (c, h) row.
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < imgH; h++) {
                const float* sr = sI + (c * imgH + h) * imgW;
                float*       dr = dI + (c * imgH + h) * imgW;
                for (size_t x = 0; x < imgW; x++) {
                    dr[x] = sr[imgW - 1 - x];
                }
            }
        }
        // Target: reverse gW per (perCell, gH) row.
        for (size_t pc = 0; pc < perCell; pc++) {
            for (size_t gh = 0; gh < gridH; gh++) {
                const float* sr = sT + (pc * gridH + gh) * gridW;
                float*       dr = dT + (pc * gridH + gh) * gridW;
                for (size_t gx = 0; gx < gridW; gx++) {
                    dr[gx] = sr[gridW - 1 - gx];
                }
            }
        }
        // Mask: reverse gW per gH row.
        for (size_t gh = 0; gh < gridH; gh++) {
            const float* sr = sM + gh * gridW;
            float*       dr = dM + gh * gridW;
            for (size_t gx = 0; gx < gridW; gx++) {
                dr[gx] = sr[gridW - 1 - gx];
            }
        }
        // x_cell (channel 0) correction: where mask=1, replace x with 1-x.
        // Channel 0 sits at offset pc=0 in the target NCHW layout.
        for (size_t gh = 0; gh < gridH; gh++) {
            for (size_t gx = 0; gx < gridW; gx++) {
                size_t cell_off = gh * gridW + gx;
                if (dM[cell_off] > 0.5f) {
                    dT[cell_off] = 1.0f - dT[cell_off];
                }
            }
        }
    }

    lean_object* inner_pair = lean_alloc_ctor(0, 2, 0);  // (target, mask)
    lean_ctor_set(inner_pair, 0, out_tgt_ba);
    lean_ctor_set(inner_pair, 1, out_msk_ba);
    lean_object* outer = lean_alloc_ctor(0, 2, 0);       // (image, (target, mask))
    lean_ctor_set(outer, 0, out_img_ba);
    lean_ctor_set(outer, 1, inner_pair);
    return lean_io_result_mk_ok(outer);
}

// Split a YOLOv1 batch slice into separately-contiguous target and mask
// tensors. Per-record layout (7200 bytes, Phase 3b format): target (5880)
// then mask (196) then numBoxes (4) then raw_boxes (1120). This helper
// only extracts target + mask (the first 6076 bytes of each record) and
// discards the bbox tail. Use yoloAugment when you need the bbox tail.
//
// Output: target_concat (batch*5880 bytes), mask_concat (batch*196 bytes).
LEAN_EXPORT lean_obj_res lean_voc_split_batch(
    b_lean_obj_arg interleaved_ba, size_t batch, lean_obj_arg w) {
    (void)w;
    const size_t target_bytes = 30 * 7 * 7 * 4;          // 5880
    const size_t mask_bytes   = 7 * 7 * 4;               // 196
    const size_t rec_bytes    = target_bytes + mask_bytes + 4 + 56 * 20;  // 7200
    size_t total_t = batch * target_bytes;
    size_t total_m = batch * mask_bytes;
    lean_object* target_ba = lean_alloc_sarray(1, total_t, total_t);
    lean_object* mask_ba   = lean_alloc_sarray(1, total_m, total_m);
    const uint8_t* src = (const uint8_t*)lean_sarray_cptr(interleaved_ba);
    uint8_t* tdst = lean_sarray_cptr(target_ba);
    uint8_t* mdst = lean_sarray_cptr(mask_ba);
    for (size_t i = 0; i < batch; i++) {
        memcpy(tdst + i * target_bytes, src + i * rec_bytes, target_bytes);
        memcpy(mdst + i * mask_bytes,   src + i * rec_bytes + target_bytes, mask_bytes);
    }
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, target_ba);
    lean_ctor_set(pair, 1, mask_ba);
    return lean_io_result_mk_ok(pair);
}

// ============================================================
// DDPM noise plumbing
// ============================================================
// Forward noising:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
// `cosine_schedule` precomputes ᾱ once; `step_inputs` is called per
// training batch and produces (x_t, ε, t) — the model is trained to
// predict ε from x_t (per-pixel MSE, useDdpm codegen branch).

// Cosine ᾱ schedule (Nichol & Dhariwal 2021). Returns a [T] f32
// ByteArray of ᾱ_t with s = 0.008. ᾱ_t = cos²((t/T+s)/(1+s) · π/2)
// normalized so ᾱ_0 = 1. Clamped to [1e-4, 0.9999] so the sampler's
// `√ᾱ_t / √ᾱ_{t-1}` ratio stays bounded near t = T (where the
// unclamped ᾱ would underflow to ~0).
LEAN_EXPORT lean_obj_res lean_ddpm_cosine_schedule(size_t T, lean_obj_arg w) {
    (void)w;
    size_t nbytes = T * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    const double s = 0.008;
    const double half_pi = 1.5707963267948966;
    double f0_arg = (0.0 / (double)T + s) / (1.0 + s) * half_pi;
    double f0 = cos(f0_arg); f0 = f0 * f0;
    for (size_t t = 0; t < T; t++) {
        double arg = ((double)t / (double)T + s) / (1.0 + s) * half_pi;
        double c = cos(arg);
        double abar = (c * c) / f0;
        if (abar < 1e-4) abar = 1e-4;
        if (abar > 0.9999) abar = 0.9999;
        o[t] = (float)abar;
    }
    return lean_io_result_mk_ok(out);
}

// xorshift64 inline for the rest of this section.
static inline uint64_t f32_xs64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

// Per training step: sample t per image, sample ε, compute x_t.
// Returns Prod ByteArray (Prod ByteArray ByteArray) — i.e. (x_t, ε, t).
LEAN_EXPORT lean_obj_res lean_ddpm_step_inputs(
    b_lean_obj_arg x0_ba, b_lean_obj_arg alphaBar_ba,
    size_t B, size_t npixels, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t T = lean_sarray_size(alphaBar_ba) / 4;
    size_t total = B * npixels;
    size_t nb_total = total * 4;
    lean_object* xt = lean_alloc_sarray(1, nb_total, nb_total);
    lean_object* eps = lean_alloc_sarray(1, nb_total, nb_total);
    lean_object* tba = lean_alloc_sarray(1, B * 4, B * 4);
    const float* x0 = (const float*)lean_sarray_cptr(x0_ba);
    const float* abar = (const float*)lean_sarray_cptr(alphaBar_ba);
    float* xtp = (float*)lean_sarray_cptr(xt);
    float* epsp = (float*)lean_sarray_cptr(eps);
    int32_t* tp = (int32_t*)lean_sarray_cptr(tba);
    uint64_t s = (uint64_t)seed ^ 0xddcc1f7e9c3a4ULL; if (s == 0) s = 1;
    for (size_t b = 0; b < B; b++) {
        // Sample timestep
        size_t t = (size_t)(f32_xs64(&s) % T);
        tp[b] = (int32_t)t;
        double abar_t = (double)abar[t];
        double sq_a = sqrt(abar_t);
        double sq_b = sqrt(1.0 - abar_t);
        // Sample ε ~ N(0, 1) for this image's npixels via Box–Muller, two at a time.
        size_t base = b * npixels;
        for (size_t i = 0; i < npixels; i += 2) {
            uint64_t r1 = f32_xs64(&s);
            uint64_t r2 = f32_xs64(&s);
            double u1 = (double)(r1 >> 11) / (double)(1ULL << 53);
            double u2 = (double)(r2 >> 11) / (double)(1ULL << 53);
            if (u1 < 1e-12) u1 = 1e-12;
            double r = sqrt(-2.0 * log(u1));
            double ang = 2.0 * 3.14159265358979323846 * u2;
            double e0 = r * cos(ang);
            double e1 = r * sin(ang);
            epsp[base + i] = (float)e0;
            xtp[base + i] = (float)(sq_a * x0[base + i] + sq_b * e0);
            if (i + 1 < npixels) {
                epsp[base + i + 1] = (float)e1;
                xtp[base + i + 1] = (float)(sq_a * x0[base + i + 1] + sq_b * e1);
            }
        }
    }
    // Pack as Prod ByteArray (Prod ByteArray ByteArray): (x_t, (ε, t))
    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, eps);
    lean_ctor_set(inner, 1, tba);
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, xt);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

// ---- Affine map of every element: out[i] = scale · in[i] + shift ----
// One pure-elementwise pass; common uses are centering [0,1] data to
// [-1,1] (scale=2, shift=-1) and the inverse for rendering (scale=0.5,
// shift=0.5).
LEAN_EXPORT lean_obj_res lean_f32_scale_shift(
    b_lean_obj_arg ba, double scale, double shift, lean_obj_arg w) {
    (void)w;
    size_t n = lean_sarray_size(ba) / 4;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(ba);
    float* o = (float*)lean_sarray_cptr(out);
    float s = (float)scale; float sh = (float)shift;
    for (size_t i = 0; i < n; i++) o[i] = s * in[i] + sh;
    return lean_io_result_mk_ok(out);
}

// ---- Prepend a constant t-channel to each image ----
// Input:  xt   [B, C*H*W] f32 (the noised image, C channels)
//         t_ba [B]        int32 (per-image timestep ∈ [0, T_max))
// Output: [B, C+1, H, W] f32 packed as a flat array, where channels
// 0..C-1 are the input image and channel C is filled with t[i]/T_max.
// Used to feed the timestep into the UNet as an extra input channel
// without a new codegen primitive.
LEAN_EXPORT lean_obj_res lean_ddpm_prepend_t_channel(
    b_lean_obj_arg xt_ba, b_lean_obj_arg t_ba,
    size_t B, size_t C, size_t H, size_t W, size_t T_max, lean_obj_arg w) {
    (void)w;
    size_t hw = H * W;
    size_t img_floats = C * hw;
    size_t per_out = (C + 1) * hw;
    size_t nbytes = B * per_out * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* xt = (const float*)lean_sarray_cptr(xt_ba);
    const int32_t* t = (const int32_t*)lean_sarray_cptr(t_ba);
    float* o = (float*)lean_sarray_cptr(out);
    float invT = 1.0f / (float)T_max;
    for (size_t i = 0; i < B; i++) {
        float* base = o + i * per_out;
        memcpy(base, xt + i * img_floats, img_floats * 4);
        float tn = ((float)t[i]) * invT;
        for (size_t k = 0; k < hw; k++) base[img_floats + k] = tn;
    }
    return lean_io_result_mk_ok(out);
}

// ---- Sinusoidal time embedding, prepended as multiple spatial channels ----
// Replaces the single-channel `t/T_max` tile with `2 * n_freq` channels
// of [sin(t · ω_k), cos(t · ω_k)] at log-spaced frequencies, broadcast
// spatially. This gives the model frequency information at multiple
// scales — the standard "Transformer/NeRF-style" positional embedding
// applied to the diffusion timestep.
//
// Frequencies follow the Vaswani 2017 / Mildenhall 2020 convention:
//     ω_k = 1 / max_period^(2k / (2 * n_freq))   for k = 0..n_freq-1
// max_period defaults to T_max (1000) so the largest period is the
// full diffusion horizon.
//
// Channel layout (output [B, C+2*n_freq, H, W]):
//     channels [0..C)        = original image
//     channels [C..C+n_freq) = sin(t · ω_k) for k = 0..n_freq-1
//     channels [C+n_freq..)  = cos(t · ω_k) for k = 0..n_freq-1
//
// Caller picks `n_freq`; 4 frequencies → 8 t-channels is a reasonable
// default for CIFAR-scale diffusion.
LEAN_EXPORT lean_obj_res lean_ddpm_prepend_sincos_t(
    b_lean_obj_arg xt_ba, b_lean_obj_arg t_ba,
    size_t B, size_t C, size_t H, size_t W,
    size_t n_freq, size_t T_max, lean_obj_arg w) {
    (void)w;
    size_t hw = H * W;
    size_t img_floats = C * hw;
    size_t out_C = C + 2 * n_freq;
    size_t per_out = out_C * hw;
    size_t nbytes = B * per_out * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* xt = (const float*)lean_sarray_cptr(xt_ba);
    const int32_t* t = (const int32_t*)lean_sarray_cptr(t_ba);
    float* o = (float*)lean_sarray_cptr(out);
    double max_period = (double)T_max;
    double inv_n = 1.0 / (double)(2 * n_freq);
    for (size_t i = 0; i < B; i++) {
        float* base = o + i * per_out;
        memcpy(base, xt + i * img_floats, img_floats * 4);
        double tv = (double)t[i];
        for (size_t k = 0; k < n_freq; k++) {
            double omega = 1.0 / pow(max_period, (double)(2 * k) * inv_n);
            float s = (float)sin(tv * omega);
            float c = (float)cos(tv * omega);
            float* sin_ch = base + img_floats + k * hw;
            float* cos_ch = base + img_floats + (n_freq + k) * hw;
            for (size_t p = 0; p < hw; p++) sin_ch[p] = s;
            for (size_t p = 0; p < hw; p++) cos_ch[p] = c;
        }
    }
    return lean_io_result_mk_ok(out);
}

// Scalar variant: `t` is a single timestep, broadcast to all images.
// Used by the sampler.
LEAN_EXPORT lean_obj_res lean_ddpm_prepend_sincos_t_scalar(
    b_lean_obj_arg xt_ba,
    size_t B, size_t C, size_t H, size_t W,
    size_t t, size_t n_freq, size_t T_max, lean_obj_arg w) {
    (void)w;
    size_t hw = H * W;
    size_t img_floats = C * hw;
    size_t out_C = C + 2 * n_freq;
    size_t per_out = out_C * hw;
    size_t nbytes = B * per_out * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* xt = (const float*)lean_sarray_cptr(xt_ba);
    float* o = (float*)lean_sarray_cptr(out);
    double max_period = (double)T_max;
    double inv_n = 1.0 / (double)(2 * n_freq);
    double tv = (double)t;
    // Precompute the sin/cos values once (same for all images).
    float* sins = (float*)malloc(n_freq * sizeof(float));
    float* coss = (float*)malloc(n_freq * sizeof(float));
    for (size_t k = 0; k < n_freq; k++) {
        double omega = 1.0 / pow(max_period, (double)(2 * k) * inv_n);
        sins[k] = (float)sin(tv * omega);
        coss[k] = (float)cos(tv * omega);
    }
    for (size_t i = 0; i < B; i++) {
        float* base = o + i * per_out;
        memcpy(base, xt + i * img_floats, img_floats * 4);
        for (size_t k = 0; k < n_freq; k++) {
            float* sin_ch = base + img_floats + k * hw;
            float* cos_ch = base + img_floats + (n_freq + k) * hw;
            for (size_t p = 0; p < hw; p++) sin_ch[p] = sins[k];
            for (size_t p = 0; p < hw; p++) cos_ch[p] = coss[k];
        }
    }
    free(sins); free(coss);
    return lean_io_result_mk_ok(out);
}

// Same as above but `t` is a single scalar broadcast to all images.
// Caller (sampler) has one timestep per DDIM step rather than a [B]
// vector.
LEAN_EXPORT lean_obj_res lean_ddpm_prepend_t_channel_scalar(
    b_lean_obj_arg xt_ba,
    size_t B, size_t C, size_t H, size_t W, size_t t, size_t T_max, lean_obj_arg w) {
    (void)w;
    size_t hw = H * W;
    size_t img_floats = C * hw;
    size_t per_out = (C + 1) * hw;
    size_t nbytes = B * per_out * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* xt = (const float*)lean_sarray_cptr(xt_ba);
    float* o = (float*)lean_sarray_cptr(out);
    float tn = (float)t / (float)T_max;
    for (size_t i = 0; i < B; i++) {
        float* base = o + i * per_out;
        memcpy(base, xt + i * img_floats, img_floats * 4);
        for (size_t k = 0; k < hw; k++) base[img_floats + k] = tn;
    }
    return lean_io_result_mk_ok(out);
}

// ---- DDIM update step (deterministic, η=0) ----
//   x_0_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
//   x_{t-1}  = √ᾱ_{t-1}·x_0_pred + √(1-ᾱ_{t-1})·ε
// Algebra: x_{t-1} = a·x_t + b·ε  with
//   a = √ᾱ_{t-1} / √ᾱ_t
//   b = √(1-ᾱ_{t-1}) - a·√(1-ᾱ_t)
// One pure-elementwise pass; caller passes a, b precomputed.
LEAN_EXPORT lean_obj_res lean_ddim_step(
    b_lean_obj_arg xt_ba, b_lean_obj_arg eps_ba,
    double a, double b, size_t n, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* xt = (const float*)lean_sarray_cptr(xt_ba);
    const float* eps = (const float*)lean_sarray_cptr(eps_ba);
    float* o = (float*)lean_sarray_cptr(out);
    float af = (float)a; float bf = (float)b;
    for (size_t i = 0; i < n; i++) o[i] = af * xt[i] + bf * eps[i];
    return lean_io_result_mk_ok(out);
}

// ---- N(0, 1) sample of `n` floats via Box–Muller ----
// Used by the sampler at inference time (training noise goes through
// `step_inputs` above). Two normals per Box–Muller iteration; for
// odd `n` we keep one of the two.
LEAN_EXPORT lean_obj_res lean_ddpm_sample_noise(size_t n, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    uint64_t s = (uint64_t)seed ^ 0xddcc1f7e9c3a4ULL; if (s == 0) s = 1;
    for (size_t i = 0; i < n; i += 2) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u1 = (double)(s >> 11) / (double)(1ULL << 53);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u2 = (double)(s >> 11) / (double)(1ULL << 53);
        if (u1 < 1e-12) u1 = 1e-12;
        double r = sqrt(-2.0 * log(u1));
        double a = 2.0 * 3.14159265358979323846 * u2;
        o[i] = (float)(r * cos(a));
        if (i + 1 < n) o[i + 1] = (float)(r * sin(a));
    }
    return lean_io_result_mk_ok(out);
}

// ---- segmentation confusion matrix (per-batch) ----
// logits : f32 [B, NC, H, W] (NCHW).  masks : u8 [B, H, W] (per-pixel
// class in 0..NC-1).  Returns int64 LE [NC*NC] confusion counts,
// conf[true*NC + pred], accumulated over the batch. The caller sums
// these across batches in exact Nat and derives per-class IoU =
// conf[c][c] / (row_c + col_c - conf[c][c]). planning/unet_demo_v2.md A.
LEAN_EXPORT lean_obj_res lean_f32_seg_confusion(
    b_lean_obj_arg logits, b_lean_obj_arg masks,
    size_t B, size_t NC, size_t H, size_t W, lean_obj_arg w_) {
    (void)w_;
    const float* lg = (const float*)lean_sarray_cptr(logits);
    const uint8_t* mk = (const uint8_t*)lean_sarray_cptr(masks);
    size_t plane = H * W;
    size_t imgsz = NC * plane;
    size_t out_n = NC * NC;
    lean_object* out = lean_alloc_sarray(1, out_n * 8, out_n * 8);
    int64_t* conf = (int64_t*)lean_sarray_cptr(out);
    for (size_t i = 0; i < out_n; i++) conf[i] = 0;
    for (size_t b = 0; b < B; b++) {
        const float* lb = lg + b * imgsz;
        const uint8_t* mb = mk + b * plane;
        for (size_t p = 0; p < plane; p++) {
            size_t best = 0;
            float bv = lb[p];
            for (size_t c = 1; c < NC; c++) {
                float v = lb[c * plane + p];
                if (v > bv) { bv = v; best = c; }
            }
            size_t t = mb[p];
            if (t < NC) conf[t * NC + best]++;
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- per-image horizontal flip for an NCHW f32 batch ----
// Each image gets an independent p=0.5 coin (xorshift64 seeded by `seed`);
// when it comes up, the W axis is reversed for every channel/row. Plain
// image aug for unconditional DDPM (no boxes/masks to co-transform) —
// planning/ddpm_demo_v2.md Workstream B3.
LEAN_EXPORT lean_obj_res lean_f32_hflip_nchw(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t H, size_t W, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t nbytes = lean_sarray_size(images);
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(images);
    float* o = (float*)lean_sarray_cptr(out);
    uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ULL;
    size_t plane = H * W;
    size_t imgsz = channels * plane;
    for (size_t b = 0; b < batch; b++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        int flip = (s & 1);
        const float* ib = in + b * imgsz;
        float* ob = o + b * imgsz;
        for (size_t c = 0; c < channels; c++) {
            for (size_t y = 0; y < H; y++) {
                const float* irow = ib + c * plane + y * W;
                float* orow = ob + c * plane + y * W;
                for (size_t x = 0; x < W; x++) {
                    orow[x] = flip ? irow[W - 1 - x] : irow[x];
                }
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- int32 LE token IDs → f32 LE ByteArray (element for element) ----
// Feeds the in-graph one-hot embedding path (tokenPositionEmbed with
// idsInput = true): the model input becomes [B, T] f32 ids and the
// one-hot is built inside the MLIR, so the host never materializes a
// [B, V*T] buffer. f32 is exact for ids < 2^24 — far past any vocab here.
LEAN_EXPORT lean_obj_res lean_f32_ids_to_floats(
    b_lean_obj_arg ids_i32, lean_obj_arg w) {
    (void)w;
    size_t nbytes = lean_sarray_size(ids_i32);
    size_t n = nbytes / 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const int32_t* in = (const int32_t*)lean_sarray_cptr(ids_i32);
    float* o = (float*)lean_sarray_cptr(out);
    for (size_t i = 0; i < n; i++) {
        o[i] = (float)in[i];
    }
    return lean_io_result_mk_ok(out);
}

// ---- uint8 mask → int32 LE mask ByteArray ----
// Pets `loadPets` returns per-pixel class labels packed as one byte per pixel.
// `trainStepAdamF32Seg` expects an int32 LE buffer (one 4-byte little-endian
// signed int per pixel, matching the classification label convention used
// elsewhere in the project). Output buffer is exactly 4× the input size.
LEAN_EXPORT lean_obj_res lean_f32_mask_u8_to_i32(
    b_lean_obj_arg mask_u8, lean_obj_arg w) {
    (void)w;
    size_t n = lean_sarray_size(mask_u8);
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const uint8_t* in = (const uint8_t*)lean_sarray_cptr(mask_u8);
    uint8_t* o = lean_sarray_cptr(out);
    for (size_t i = 0; i < n; i++) {
        o[i*4]     = in[i];
        o[i*4 + 1] = 0;
        o[i*4 + 2] = 0;
        o[i*4 + 3] = 0;
    }
    return lean_io_result_mk_ok(out);
}

// ---- Shuffle images + labels in-place (Fisher-Yates) ----
// images: n * pixels_per * 4 bytes; labels: n * 4 bytes
LEAN_EXPORT lean_obj_res lean_f32_shuffle(lean_obj_arg img_obj, lean_obj_arg lbl_obj,
                                          size_t n, size_t pixels_per, size_t seed,
                                          lean_obj_arg w) {
    (void)w;
    // Ensure exclusive ownership (rc == 1) for in-place mutation
    if (!lean_is_exclusive(img_obj)) img_obj = lean_copy_byte_array(img_obj);
    if (!lean_is_exclusive(lbl_obj)) lbl_obj = lean_copy_byte_array(lbl_obj);
    uint8_t* img = lean_sarray_cptr(img_obj);
    uint8_t* lbl = lean_sarray_cptr(lbl_obj);
    size_t img_stride = pixels_per * 4;
    // Temp buffer for one image
    uint8_t* tmp = (uint8_t*)malloc(img_stride > 4 ? img_stride : 4);
    uint64_t rng = seed ^ 0x5DEECE66DUL;
    for (size_t i = n - 1; i > 0; i--) {
        rng = rng * 6364136223846793005UL + 1442695040888963407UL;
        size_t j = (size_t)((rng >> 16) % (i + 1));
        if (i != j) {
            // Swap images
            memcpy(tmp, img + i * img_stride, img_stride);
            memcpy(img + i * img_stride, img + j * img_stride, img_stride);
            memcpy(img + j * img_stride, tmp, img_stride);
            // Swap labels
            memcpy(tmp, lbl + i * 4, 4);
            memcpy(lbl + i * 4, lbl + j * 4, 4);
            memcpy(lbl + j * 4, tmp, 4);
        }
    }
    free(tmp);
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, img_obj);
    lean_ctor_set(pair, 1, lbl_obj);
    return lean_io_result_mk_ok(pair);
}

// ---- EMA update: running = (1-momentum)*running + momentum*batch ----
LEAN_EXPORT lean_obj_res lean_f32_ema(
    b_lean_obj_arg running_ba, b_lean_obj_arg batch_ba,
    double momentum, lean_obj_arg w) {
    (void)w;
    size_t n = lean_sarray_size(running_ba) / 4;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* r = (const float*)lean_sarray_cptr(running_ba);
    const float* b = (const float*)lean_sarray_cptr(batch_ba);
    float* o = (float*)lean_sarray_cptr(out);
    float mom = (float)momentum;
    float omom = 1.0f - mom;
    for (size_t i = 0; i < n; i++) o[i] = omom * r[i] + mom * b[i];
    return lean_io_result_mk_ok(out);
}

// ---- Random crop: batch of NCHW images from src_size to crop_size ----
// Input: batch * C * src_h * src_w floats (already normalized).
// Output: batch * C * crop_h * crop_w floats (random offset per image).
LEAN_EXPORT lean_obj_res lean_f32_random_crop(
    b_lean_obj_arg ba, size_t batch, size_t channels,
    size_t src_h, size_t src_w, size_t crop_h, size_t crop_w,
    size_t seed, lean_obj_arg w) {
    (void)w;
    size_t out_pixels = channels * crop_h * crop_w;
    size_t out_nbytes = batch * out_pixels * 4;
    size_t src_pixels = channels * src_h * src_w;
    lean_object* out = lean_alloc_sarray(1, out_nbytes, out_nbytes);
    float* dst = (float*)lean_sarray_cptr(out);
    const float* src = (const float*)lean_sarray_cptr(ba);

    size_t max_y = src_h - crop_h;
    size_t max_x = src_w - crop_w;
    uint64_t s = seed + 1;
    for (size_t i = 0; i < batch; i++) {
        // xorshift64 for random offsets
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t y0 = (max_y > 0) ? (s % (max_y + 1)) : 0;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t x0 = (max_x > 0) ? (s % (max_x + 1)) : 0;

        const float* img_src = src + i * src_pixels;
        float* img_dst = dst + i * out_pixels;
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_h; h++) {
                memcpy(img_dst + c * crop_h * crop_w + h * crop_w,
                       img_src + c * src_h * src_w + (y0 + h) * src_w + x0,
                       crop_w * sizeof(float));
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- Deterministic center crop for a batch of NCHW images ----
// Same shape as random_crop but y0/x0 are fixed to (max/2) — same crop
// window for every image, no RNG. Used as the no-augment fallback for
// Imagenette (stored 256x256, model input 224x224) so training gets the
// expected tensor shape even when cfg.augment=false.
LEAN_EXPORT lean_obj_res lean_f32_center_crop(
    b_lean_obj_arg ba, size_t batch, size_t channels,
    size_t src_h, size_t src_w, size_t crop_h, size_t crop_w,
    lean_obj_arg w) {
    (void)w;
    size_t out_pixels = channels * crop_h * crop_w;
    size_t out_nbytes = batch * out_pixels * 4;
    size_t src_pixels = channels * src_h * src_w;
    lean_object* out = lean_alloc_sarray(1, out_nbytes, out_nbytes);
    float* dst = (float*)lean_sarray_cptr(out);
    const float* src = (const float*)lean_sarray_cptr(ba);

    size_t y0 = (src_h > crop_h) ? (src_h - crop_h) / 2 : 0;
    size_t x0 = (src_w > crop_w) ? (src_w - crop_w) / 2 : 0;
    for (size_t i = 0; i < batch; i++) {
        const float* img_src = src + i * src_pixels;
        float* img_dst = dst + i * out_pixels;
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_h; h++) {
                memcpy(img_dst + c * crop_h * crop_w + h * crop_w,
                       img_src + c * src_h * src_w + (y0 + h) * src_w + x0,
                       crop_w * sizeof(float));
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- Random horizontal flip for a batch of NCHW images (in-place on copy) ----
// pixels_per_image = C * H * W, width = W, 50% chance per image.
LEAN_EXPORT lean_obj_res lean_f32_random_hflip(
    b_lean_obj_arg ba, size_t batch, size_t channels,
    size_t height, size_t width, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels_per_image = channels * height * width;
    size_t nbytes = batch * pixels_per_image * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    memcpy(lean_sarray_cptr(out), lean_sarray_cptr(ba), nbytes);
    float* data = (float*)lean_sarray_cptr(out);

    uint64_t s = seed + 1;
    for (size_t i = 0; i < batch; i++) {
        // xorshift64 for random decision
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        if (s & 1) {
            // Flip this image horizontally: reverse each row in each channel
            float* img = data + i * pixels_per_image;
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    float* row = img + c * height * width + h * width;
                    for (size_t l = 0, r = width - 1; l < r; l++, r--) {
                        float tmp = row[l];
                        row[l] = row[r];
                        row[r] = tmp;
                    }
                }
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// Paired horizontal flip for segmentation: flips the f32 image and the uint8
// per-pixel mask TOGETHER with one coin per image, so pixel correspondence is
// preserved exactly. A hflip is a pure permutation of columns, so the mask
// needs no interpolation and no label can be invented — the reason segmentation
// aug must flip labels nearest-neighbour is exactly satisfied by a flip.
// Brains are near-symmetric, so hflip is label-safe here (it would not be for
// an organ with strong laterality). Returns (new_image, new_mask).
LEAN_EXPORT lean_obj_res lean_f32_seg_hflip_pair(
    b_lean_obj_arg img_ba, b_lean_obj_arg mask_ba,
    size_t batch, size_t channels, size_t height, size_t width,
    size_t seed, lean_obj_arg w) {
    (void)w;
    size_t img_ppi = channels * height * width;   // floats per image
    size_t mask_ppi = height * width;             // bytes per mask
    size_t img_bytes = batch * img_ppi * 4;
    size_t mask_bytes = batch * mask_ppi;
    lean_object* img_out = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* mask_out = lean_alloc_sarray(1, mask_bytes, mask_bytes);
    memcpy(lean_sarray_cptr(img_out), lean_sarray_cptr(img_ba), img_bytes);
    memcpy(lean_sarray_cptr(mask_out), lean_sarray_cptr(mask_ba), mask_bytes);
    float* idata = (float*)lean_sarray_cptr(img_out);
    uint8_t* mdata = lean_sarray_cptr(mask_out);

    uint64_t s = seed + 1;
    for (size_t i = 0; i < batch; i++) {
        // Same xorshift64 stream as lean_f32_random_hflip, so the decision is
        // reproducible; one coin drives BOTH image and mask for this image.
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        if (s & 1) {
            float* img = idata + i * img_ppi;
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    float* row = img + c * height * width + h * width;
                    for (size_t l = 0, r = width - 1; l < r; l++, r--) {
                        float tmp = row[l]; row[l] = row[r]; row[r] = tmp;
                    }
                }
            }
            uint8_t* msk = mdata + i * mask_ppi;
            for (size_t h = 0; h < height; h++) {
                uint8_t* row = msk + h * width;
                for (size_t l = 0, r = width - 1; l < r; l++, r--) {
                    uint8_t tmp = row[l]; row[l] = row[r]; row[r] = tmp;
                }
            }
        }
    }
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, img_out);
    lean_ctor_set(pair, 1, mask_out);
    return lean_io_result_mk_ok(pair);
}

// ============================================================
// Mixup / CutMix / Random Erasing — DeiT-style augmentation pack
// ============================================================
//
// Mixup (Zhang et al. 2017):
//   λ ~ Beta(α, α); pick a permutation π over the batch.
//   x_mixed[i]    = λ·x[i]       + (1-λ)·x[π(i)]
//   y_mixed[i, c] = λ·smooth(y[i], c) + (1-λ)·smooth(y[π(i)], c)
//
// CutMix (Yun et al. 2019):
//   λ ~ Beta(α, α); pick a random rectangle of size √(1-λ);
//   paste rectangle from x[π(i)] onto x[i].
//   λ_actual = 1 - (rect_area / image_area)  (bounded by image edges)
//   y_mixed[i, c] = λ_actual·smooth(y[i], c) + (1-λ_actual)·smooth(y[π(i)], c)
//
// Random Erasing (Zhong et al. 2017): with probability p,
//   pick a rectangle of relative area in [s_min, s_max] and
//   aspect ratio in [r_min, r_max], fill with N(0, 1) noise. Per image,
//   independent. Labels unchanged.
//
// Mixup and CutMix expose two FFI calls each, sharing a seed: `_images`
// computes the mixed-image tensor; `_soft_labels` recomputes λ + π from
// the same seed and emits a `[batch × n_classes]` smoothed soft-label
// tensor. Calling with the same seed in both is critical — the label
// must match the image mix.

// xorshift64 step. Returns a uint64 in (0, 2^64).
static inline uint64_t f32_xorshift64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

// Uniform(0, 1) from xorshift state.
static inline double f32_unif01(uint64_t* s) {
    uint64_t x = f32_xorshift64(s);
    return (double)(x >> 11) / (double)(1ULL << 53);
}

// Standard normal via Box-Muller (returns one of the two values).
static inline double f32_randn(uint64_t* s) {
    double u1 = f32_unif01(s);
    double u2 = f32_unif01(s);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// ---- Tile one image + add N(0, sigma^2) Gaussian noise (randomized smoothing) ----
// `base[off .. off+d0)` is ONE flattened image (d0 floats, element offset `off`).
// Returns `m*d0` floats: m independent noisy copies, copy k = image + N(0,sigma^2 I),
// each coordinate an independent Box-Muller draw from an xorshift64 stream seeded by
// `seed`. NO clipping — the smoothing certificate (Cohen–Rosenfeld–Kolter 2019) lives
// in raw input L2 space, so clipping to [0,1] would break the Gaussian isometry.
// (Used both to noise-augment a training batch — off=0, d0=bs*pix, m=1 — and to draw
//  the m certify samples for one test image.)
LEAN_EXPORT lean_obj_res lean_f32_add_gaussian_tiled(
        b_lean_obj_arg base, size_t off, size_t d0, size_t m,
        double sigma, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t n = m * d0;
    size_t nbytes = n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    const float* src = (const float*)lean_sarray_cptr(base);
    uint64_t s = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    float fsig = (float)sigma;
    for (size_t k = 0; k < m; k++) {
        for (size_t i = 0; i < d0; i++) {
            p[k * d0 + i] = src[off + i] + fsig * (float)f32_randn(&s);
        }
    }
    return lean_io_result_mk_ok(ba);
}

// ---- Perturb one image by r·(random unit vector): x + r·u, ‖r·u‖₂ = r exactly ----
// Returns `base[off .. off+d0)` shifted by a magnitude-`r` step in a uniformly-random direction
// (Box-Muller draw of u ~ N(0,I), then normalized). Used by the Lipschitz-hypothesis probe to
// measure how Φ⁻¹(P[f(x+η)=c]) moves under an input shift of known L2 size.
LEAN_EXPORT lean_obj_res lean_f32_perturb_unit(
        b_lean_obj_arg base, size_t off, size_t d0, double r, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t nbytes = d0 * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    const float* src = (const float*)lean_sarray_cptr(base);
    uint64_t s = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    double norm2 = 0.0;
    for (size_t i = 0; i < d0; i++) {
        double z = f32_randn(&s);
        p[i] = (float)z;                 // stash the raw direction
        norm2 += z * z;
    }
    double scale = (norm2 > 0.0) ? (r / sqrt(norm2)) : 0.0;
    for (size_t i = 0; i < d0; i++) {
        p[i] = src[off + i] + (float)(scale * (double)p[i]);
    }
    return lean_io_result_mk_ok(ba);
}

// Marsaglia & Tsang gamma sampler for shape α > 0, scale 1.
// For α < 1, use the boost trick: G(α) = G(α+1) · U^(1/α).
static double f32_gamma_sample(double alpha, uint64_t* s) {
    if (alpha < 1.0) {
        double g1 = f32_gamma_sample(alpha + 1.0, s);
        double u = f32_unif01(s);
        if (u < 1e-300) u = 1e-300;
        return g1 * pow(u, 1.0 / alpha);
    }
    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    while (1) {
        double x = f32_randn(s);
        double v = 1.0 + c * x;
        if (v <= 0) continue;
        v = v * v * v;
        double u = f32_unif01(s);
        if (log(u) < 0.5 * x * x + d - d * v + d * log(v)) {
            return d * v;
        }
    }
}

// Beta(α, α) via two Gammas.
static double f32_beta_symmetric(double alpha, uint64_t* s) {
    double g1 = f32_gamma_sample(alpha, s);
    double g2 = f32_gamma_sample(alpha, s);
    return g1 / (g1 + g2);
}

// Fisher-Yates permutation [0, batch). Caller provides storage.
static void f32_permutation(size_t* perm, size_t batch, uint64_t* s) {
    for (size_t i = 0; i < batch; i++) perm[i] = i;
    for (size_t i = batch - 1; i > 0; i--) {
        uint64_t r = f32_xorshift64(s);
        size_t j = (size_t)(r % (i + 1));
        size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
}

// Smoothed onehot value for label `y` at class `c`.
//   smooth = label_smoothing in [0, 1).
//   If c == y: 1 - smooth + smooth/N
//   Else:      smooth/N
static inline float f32_smooth_onehot(int y, size_t c, double smooth, size_t n_classes) {
    double off = smooth / (double)n_classes;
    if ((size_t)y == c) return (float)(1.0 - smooth + off);
    return (float)off;
}

// ----------------------------------------------------------------
// Mixup: images. Returns ByteArray of shape [batch, C, H, W] f32.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_mixup_images(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double alpha, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(images);
    float* o = (float*)lean_sarray_cptr(out);
    uint64_t s = seed ^ 0xa1b2c3d4e5f60718ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    if (lambda < 0.5) lambda = 1.0 - lambda;  // bias toward keeping main image dominant
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    float l = (float)lambda;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        const float* a = in + i * pixels;
        const float* b = in + perm[i] * pixels;
        float* d = o + i * pixels;
        for (size_t k = 0; k < pixels; k++) d[k] = l * a[k] + l1 * b[k];
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// Mixup: soft labels. Returns [batch, n_classes] f32.
// MUST use same `seed` and `alpha` as the matching mixup_images call.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_mixup_soft_labels(
    b_lean_obj_arg int_labels, size_t batch, size_t n_classes,
    double alpha, double smooth, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t out_n = batch * n_classes;
    size_t nbytes = out_n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    const uint8_t* lbl = (const uint8_t*)lean_sarray_cptr(int_labels);
    uint64_t s = seed ^ 0xa1b2c3d4e5f60718ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    if (lambda < 0.5) lambda = 1.0 - lambda;
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    float l = (float)lambda;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        int32_t y_a, y_b;
        memcpy(&y_a, lbl + i * 4, 4);
        memcpy(&y_b, lbl + perm[i] * 4, 4);
        for (size_t c = 0; c < n_classes; c++) {
            float a = f32_smooth_onehot(y_a, c, smooth, n_classes);
            float b = f32_smooth_onehot(y_b, c, smooth, n_classes);
            o[i * n_classes + c] = l * a + l1 * b;
        }
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// CutMix: images. Returns mixed-image ByteArray.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_cutmix_images(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double alpha, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(images);
    float* o = (float*)lean_sarray_cptr(out);
    memcpy(o, in, nbytes);
    uint64_t s = seed ^ 0xb2c3d4e5f6071829ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    double cut_ratio = sqrt(1.0 - lambda);
    size_t cut_h = (size_t)((double)height * cut_ratio);
    size_t cut_w = (size_t)((double)width * cut_ratio);
    if (cut_h < 1) cut_h = 1; if (cut_w < 1) cut_w = 1;
    size_t cy = (size_t)(f32_unif01(&s) * (double)height);
    size_t cx = (size_t)(f32_unif01(&s) * (double)width);
    size_t y1 = cy > cut_h / 2 ? cy - cut_h / 2 : 0;
    size_t y2 = cy + cut_h / 2; if (y2 > height) y2 = height;
    size_t x1 = cx > cut_w / 2 ? cx - cut_w / 2 : 0;
    size_t x2 = cx + cut_w / 2; if (x2 > width) x2 = width;
    for (size_t i = 0; i < batch; i++) {
        const float* b_img = in + perm[i] * pixels;
        float* d_img = o + i * pixels;
        for (size_t c = 0; c < channels; c++) {
            for (size_t y = y1; y < y2; y++) {
                size_t off = c * height * width + y * width;
                for (size_t x = x1; x < x2; x++) {
                    d_img[off + x] = b_img[off + x];
                }
            }
        }
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// CutMix: soft labels. Recomputes λ_actual from rectangle area.
// MUST use same `seed/alpha` as cutmix_images.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_cutmix_soft_labels(
    b_lean_obj_arg int_labels, size_t batch, size_t n_classes,
    size_t height, size_t width, double alpha, double smooth,
    size_t seed, lean_obj_arg w) {
    (void)w;
    size_t out_n = batch * n_classes;
    size_t nbytes = out_n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    const uint8_t* lbl = (const uint8_t*)lean_sarray_cptr(int_labels);
    uint64_t s = seed ^ 0xb2c3d4e5f6071829ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    double cut_ratio = sqrt(1.0 - lambda);
    size_t cut_h = (size_t)((double)height * cut_ratio);
    size_t cut_w = (size_t)((double)width * cut_ratio);
    if (cut_h < 1) cut_h = 1; if (cut_w < 1) cut_w = 1;
    size_t cy = (size_t)(f32_unif01(&s) * (double)height);
    size_t cx = (size_t)(f32_unif01(&s) * (double)width);
    size_t y1 = cy > cut_h / 2 ? cy - cut_h / 2 : 0;
    size_t y2 = cy + cut_h / 2; if (y2 > height) y2 = height;
    size_t x1 = cx > cut_w / 2 ? cx - cut_w / 2 : 0;
    size_t x2 = cx + cut_w / 2; if (x2 > width) x2 = width;
    double rect_area = (double)((y2 - y1) * (x2 - x1));
    double total_area = (double)(height * width);
    double l_actual = 1.0 - rect_area / total_area;
    float l = (float)l_actual;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        int32_t y_a, y_b;
        memcpy(&y_a, lbl + i * 4, 4);
        memcpy(&y_b, lbl + perm[i] * 4, 4);
        for (size_t c = 0; c < n_classes; c++) {
            float a = f32_smooth_onehot(y_a, c, smooth, n_classes);
            float b = f32_smooth_onehot(y_b, c, smooth, n_classes);
            o[i * n_classes + c] = l * a + l1 * b;
        }
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// KNN-Mixup: same as Mixup, but pair[i] is the nearest neighbor of i
// in pixel-space L2 distance instead of a random permutation. Mixes
// each sample with its closest sibling in the batch — the "manifold
// neighbor" of i — producing harder, more realistic mixed images
// than random Mixup. Pair with `knn_mixup_soft_labels` (same images,
// same seed, same alpha).
// ----------------------------------------------------------------
static void f32_knn_pairing(size_t* pair, const float* in,
                            size_t batch, size_t pixels) {
    for (size_t i = 0; i < batch; i++) {
        double best = 1e30; size_t best_j = (i + 1) % batch;
        const float* a = in + i * pixels;
        for (size_t j = 0; j < batch; j++) {
            if (j == i) continue;
            const float* b = in + j * pixels;
            double d = 0.0;
            for (size_t k = 0; k < pixels; k++) {
                double v = (double)a[k] - (double)b[k];
                d += v * v;
                if (d > best) break;  // early termination
            }
            if (d < best) { best = d; best_j = j; }
        }
        pair[i] = best_j;
    }
}

LEAN_EXPORT lean_obj_res lean_f32_knn_mixup_images(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double alpha, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(images);
    float* o = (float*)lean_sarray_cptr(out);
    uint64_t s = seed ^ 0xa1b2c3d4e5f60718ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    if (lambda < 0.5) lambda = 1.0 - lambda;
    size_t* pair = (size_t*)malloc(batch * sizeof(size_t));
    f32_knn_pairing(pair, in, batch, pixels);
    float l = (float)lambda;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        const float* a = in + i * pixels;
        const float* b = in + pair[i] * pixels;
        float* d = o + i * pixels;
        for (size_t k = 0; k < pixels; k++) d[k] = l * a[k] + l1 * b[k];
    }
    free(pair);
    return lean_io_result_mk_ok(out);
}

// KNN-Mixup soft labels. Needs `images` to recompute the same KNN pairing
// the matching `_images` call used. Same seed reproduces the same λ.
LEAN_EXPORT lean_obj_res lean_f32_knn_mixup_soft_labels(
    b_lean_obj_arg int_labels, b_lean_obj_arg images,
    size_t batch, size_t n_classes, size_t channels,
    size_t height, size_t width, double alpha, double smooth,
    size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t out_n = batch * n_classes;
    size_t nbytes = out_n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    const uint8_t* lbl = (const uint8_t*)lean_sarray_cptr(int_labels);
    const float* in = (const float*)lean_sarray_cptr(images);
    uint64_t s = seed ^ 0xa1b2c3d4e5f60718ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    if (lambda < 0.5) lambda = 1.0 - lambda;
    size_t* pair = (size_t*)malloc(batch * sizeof(size_t));
    f32_knn_pairing(pair, in, batch, pixels);
    float l = (float)lambda;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        int32_t y_a, y_b;
        memcpy(&y_a, lbl + i * 4, 4);
        memcpy(&y_b, lbl + pair[i] * 4, 4);
        for (size_t c = 0; c < n_classes; c++) {
            float a = f32_smooth_onehot(y_a, c, smooth, n_classes);
            float b = f32_smooth_onehot(y_b, c, smooth, n_classes);
            o[i * n_classes + c] = l * a + l1 * b;
        }
    }
    free(pair);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// EMA on squared values: out = (1-mom)*running + mom*batch².
// Used by SWAG to maintain a running E[θ²] alongside SWA's running E[θ].
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_ema_sq(
    b_lean_obj_arg running_ba, b_lean_obj_arg batch_ba,
    double momentum, lean_obj_arg w) {
    (void)w;
    size_t n = lean_sarray_size(running_ba) / 4;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* r = (const float*)lean_sarray_cptr(running_ba);
    const float* b = (const float*)lean_sarray_cptr(batch_ba);
    float* o = (float*)lean_sarray_cptr(out);
    float mom = (float)momentum;
    float omom = 1.0f - mom;
    for (size_t i = 0; i < n; i++) {
        float bi = b[i];
        o[i] = omom * r[i] + mom * (bi * bi);
    }
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// Element-wise subtract: out[i] = a[i] − b[i]. Used by SWAG to compute
// per-epoch deviations `p − swaMean`.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_subtract(
    b_lean_obj_arg a_ba, b_lean_obj_arg b_ba, lean_obj_arg w) {
    (void)w;
    size_t n = lean_sarray_size(a_ba) / 4;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* a = (const float*)lean_sarray_cptr(a_ba);
    const float* b = (const float*)lean_sarray_cptr(b_ba);
    float* o = (float*)lean_sarray_cptr(out);
    for (size_t i = 0; i < n; i++) o[i] = a[i] - b[i];
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// SWAG sample weights. Builds θ_s = swaMean + (1/√2)·√σ_diag·z₁
//                                    + (1/√(2(K−1)))·D·z₂
// where:
//   σ_diag[i] = max(swaSq[i] − swaMean[i]², 0)   (clamped diagonal variance)
//   D ∈ ℝ^{K × P} is the row-major flatten of the K most recent
//   per-epoch deviations from swaMean (one row = one snapshot).
//   z₁ ∈ ℝ^P, z₂ ∈ ℝ^K standard normal.
// Reference: Maddox, Garipov, Izmailov, Vetrov, Wilson 2019.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_swag_sample(
    b_lean_obj_arg swa_mean, b_lean_obj_arg swa_sq,
    b_lean_obj_arg deviations,
    size_t n_params, size_t k, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n_params * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* mu = (const float*)lean_sarray_cptr(swa_mean);
    const float* sq = (const float*)lean_sarray_cptr(swa_sq);
    const float* d = (const float*)lean_sarray_cptr(deviations);
    float* o = (float*)lean_sarray_cptr(out);
    uint64_t s = seed ^ 0xd4e5f60718293a4bULL; if (s == 0) s = 1;
    // Pre-sample z₂ ∈ ℝ^k (small, fits in stack).
    float* z2 = (float*)malloc(k * sizeof(float));
    for (size_t j = 0; j < k; j++) z2[j] = (float)f32_randn(&s);
    float scale_low = (k > 1) ? (float)(1.0 / sqrt(2.0 * (double)(k - 1))) : 0.0f;
    float scale_diag = (float)(1.0 / sqrt(2.0));
    for (size_t i = 0; i < n_params; i++) {
        float mean_i = mu[i];
        float var_i = sq[i] - mean_i * mean_i;
        if (var_i < 0.0f) var_i = 0.0f;
        float z1 = (float)f32_randn(&s);
        float low_dot = 0.0f;
        for (size_t j = 0; j < k; j++) {
            // d[j * n_params + i] = j-th deviation, parameter i.
            low_dot += d[j * n_params + i] * z2[j];
        }
        o[i] = mean_i + scale_diag * sqrtf(var_i) * z1 + scale_low * low_dot;
    }
    free(z2);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// Random Erasing. Per-image, with probability `prob`, fills a random
// rectangle (area 2-33% of image, aspect ratio 0.3-3.3) with N(0,1)
// noise. Labels unchanged — caller can keep using int32 labels.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_random_erasing(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double prob, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    memcpy(o, lean_sarray_cptr(images), nbytes);
    uint64_t s = seed ^ 0xc3d4e5f60718293aULL; if (s == 0) s = 1;
    const double s_min = 0.02, s_max = 0.33;
    const double r_min = 0.3, r_max = 3.3;
    for (size_t i = 0; i < batch; i++) {
        if (f32_unif01(&s) >= prob) continue;
        for (int attempt = 0; attempt < 10; attempt++) {
            double area = (double)(height * width);
            double target_area = area * (s_min + (s_max - s_min) * f32_unif01(&s));
            double aspect = exp(log(r_min) + (log(r_max) - log(r_min)) * f32_unif01(&s));
            size_t h_e = (size_t)round(sqrt(target_area * aspect));
            size_t w_e = (size_t)round(sqrt(target_area / aspect));
            if (h_e >= height || w_e >= width || h_e < 1 || w_e < 1) continue;
            size_t y0 = (size_t)(f32_unif01(&s) * (double)(height - h_e));
            size_t x0 = (size_t)(f32_unif01(&s) * (double)(width - w_e));
            for (size_t c = 0; c < channels; c++) {
                for (size_t y = y0; y < y0 + h_e; y++) {
                    size_t off = i * pixels + c * height * width + y * width;
                    for (size_t x = x0; x < x0 + w_e; x++) {
                        o[off + x] = (float)f32_randn(&s);
                    }
                }
            }
            break;
        }
    }
    return lean_io_result_mk_ok(out);
}

// ============================================================
// RandAugment-Color (Cubuk et al. 2019, color-only subset)
// ============================================================
// Implements 4 color ops (brightness, contrast, color, autocontrast)
// + identity, applied to images in [0, 1] sRGB space:
//
//   identity      — no-op
//   brightness    — img *= factor                  (factor ~ 1 ± strength*0.5)
//   contrast      — lerp pixels around per-image mean  (same factor range)
//   color         — lerp toward per-pixel grayscale (saturation knob)
//   autocontrast  — stretch min→0 / max→1 (no magnitude, image-derived)
//
// `imagenet_norm` flag tells the kernel whether the incoming images
// are ImageNet-mean/std normalized (Imagenette). When set, we de-norm
// to [0,1] sRGB at the start of each per-image augmentation pass,
// apply ops in [0,1] (with clamp between ops), then re-norm at the
// end. CIFAR / MNIST / pre-normalized data passes flag=0 and the ops
// apply directly. Geometric ops (rotate/shear/translate) still TODO.
// ----------------------------------------------------------------

static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// De-normalize NCHW (channels=3, ImageNet stats): x = x_norm*std + mean.
static inline void denormalize_imagenet(float* img, size_t channels,
                                        size_t plane) {
    if (channels != 3) return;
    for (size_t c = 0; c < channels; c++) {
        float* p = img + c * plane;
        float mu = IMAGENET_MEAN[c], sd = IMAGENET_STD[c];
        for (size_t k = 0; k < plane; k++) p[k] = p[k] * sd + mu;
    }
}

// Re-normalize NCHW (channels=3, ImageNet stats): x_norm = (x - mean)/std.
static inline void renormalize_imagenet(float* img, size_t channels,
                                        size_t plane) {
    if (channels != 3) return;
    for (size_t c = 0; c < channels; c++) {
        float* p = img + c * plane;
        float mu = IMAGENET_MEAN[c], inv_sd = 1.0f / IMAGENET_STD[c];
        for (size_t k = 0; k < plane; k++) p[k] = (p[k] - mu) * inv_sd;
    }
}

// Clamp to [0,1] — used between ops to keep values in valid sRGB range.
static inline void clamp01(float* img, size_t pixels) {
    for (size_t k = 0; k < pixels; k++) {
        if (img[k] < 0.0f) img[k] = 0.0f;
        else if (img[k] > 1.0f) img[k] = 1.0f;
    }
}

static inline void apply_brightness(float* img, size_t pixels, double factor) {
    float f = (float)factor;
    for (size_t k = 0; k < pixels; k++) img[k] *= f;
}

static inline void apply_contrast(float* img, size_t pixels, double factor) {
    double mean = 0.0;
    for (size_t k = 0; k < pixels; k++) mean += img[k];
    mean /= (double)pixels;
    float m = (float)mean, f = (float)factor;
    for (size_t k = 0; k < pixels; k++) img[k] = m + f * (img[k] - m);
}

// NCHW with channels=3 only. mag=1 → identity, mag=0 → grayscale.
static inline void apply_color(float* img, size_t channels, size_t height,
                               size_t width, double factor) {
    if (channels != 3) return;
    size_t plane = height * width;
    float f = (float)factor, om = 1.0f - f;
    float* r = img;
    float* g = img + plane;
    float* b = img + 2 * plane;
    for (size_t k = 0; k < plane; k++) {
        float gray = 0.299f * r[k] + 0.587f * g[k] + 0.114f * b[k];
        r[k] = f * r[k] + om * gray;
        g[k] = f * g[k] + om * gray;
        b[k] = f * b[k] + om * gray;
    }
}

static inline void apply_autocontrast(float* img, size_t pixels) {
    float mn = img[0], mx = img[0];
    for (size_t k = 1; k < pixels; k++) {
        if (img[k] < mn) mn = img[k];
        if (img[k] > mx) mx = img[k];
    }
    float range = mx - mn;
    if (range < 1e-6f) return;
    float inv = 1.0f / range;
    for (size_t k = 0; k < pixels; k++) img[k] = (img[k] - mn) * inv;
}

// Magnitude → factor: factor = 1 + sign * strength * 0.5 ∈ [0.5, 1.5] at M=10
static inline double rand_factor(double m, uint64_t* s) {
    double strength = m / 10.0;
    if (strength > 1.0) strength = 1.0;
    double sign = (f32_unif01(s) < 0.5) ? -1.0 : 1.0;
    return 1.0 + sign * strength * 0.5;
}

// ---- Token corpus loader ----
// Read a flat int32 LE token-ID file into a ByteArray. Returns
// (token_count : Nat) for the Lean side to slice from.
LEAN_EXPORT lean_obj_res lean_f32_load_token_stream(
    b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("token stream file not found")));
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes < 0 || (bytes % 4) != 0) {
        fclose(f);
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("token stream not int32-aligned")));
    }
    lean_object* ba = lean_alloc_sarray(1, (size_t)bytes, (size_t)bytes);
    size_t got = fread(lean_sarray_cptr(ba), 1, (size_t)bytes, f);
    fclose(f);
    if (got != (size_t)bytes) {
        lean_dec_ref(ba);
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("short read on token stream")));
    }
    size_t n_tokens = (size_t)bytes / 4;
    // Return (ba, n_tokens) as a Lean `(ByteArray × Nat)` pair.
    lean_object* tup = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(tup, 0, ba);
    lean_ctor_set(tup, 1, lean_box_usize(n_tokens));
    return lean_io_result_mk_ok(tup);
}

// ---- Random-chunk batch sampler for autoregressive language modeling ----
// Pick `batch` random offsets in [0, n_tokens - seq_len - 1] and copy
// (tokens[off..off+T], tokens[off+1..off+T+1]) into two int32 tensors.
// Returns (input_ids : [B*T*4 bytes], next_ids : [B*T*4 bytes]) packed
// into a single ByteArray of size 2*B*T*4 — first half input, second
// half target. The Lean caller slices.
LEAN_EXPORT lean_obj_res lean_f32_sample_chunks(
    b_lean_obj_arg tokens_ba, size_t n_tokens, size_t batch, size_t seq_len,
    size_t seed, lean_obj_arg w) {
    (void)w;
    if (n_tokens < seq_len + 1) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string("n_tokens < seq_len + 1")));
    }
    const int32_t* tokens = (const int32_t*)lean_sarray_cptr(tokens_ba);
    size_t per_chunk = seq_len * 4;
    size_t out_bytes = 2 * batch * per_chunk;
    lean_object* out = lean_alloc_sarray(1, out_bytes, out_bytes);
    int32_t* input = (int32_t*)lean_sarray_cptr(out);
    int32_t* target = input + batch * seq_len;
    uint64_t s = (uint64_t)seed ^ 0xa6b8c4d2e3f10987ULL; if (s == 0) s = 1;
    size_t max_off = n_tokens - seq_len - 1;
    for (size_t b = 0; b < batch; b++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t off = (size_t)(s % (max_off + 1));
        memcpy(input + b * seq_len, tokens + off, per_chunk);
        memcpy(target + b * seq_len, tokens + off + 1, per_chunk);
    }
    return lean_io_result_mk_ok(out);
}

// ---- One-hot encode int32 token IDs into a flat f32 tensor ----
// Input  : int32 [B*T] token IDs (raw bytes from sample_chunks).
// Output : f32 [B, T*V] flat (one-hot per position, row-major in (b, t, v)).
// V = vocab_size. The flat layout matches inputFlatDim for a tinyGPT spec
// whose first layer is the token+position embedding.
LEAN_EXPORT lean_obj_res lean_f32_token_one_hot(
    b_lean_obj_arg ids_ba, size_t batch, size_t seq_len, size_t vocab,
    lean_obj_arg w) {
    (void)w;
    const int32_t* ids = (const int32_t*)lean_sarray_cptr(ids_ba);
    size_t per_token = vocab * 4;
    size_t out_bytes = batch * seq_len * per_token;
    lean_object* out = lean_alloc_sarray(1, out_bytes, out_bytes);
    float* f = (float*)lean_sarray_cptr(out);
    memset(f, 0, out_bytes);
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int32_t id = ids[b * seq_len + t];
            if (id < 0 || (size_t)id >= vocab) continue;  // silently clamp OOV
            f[(b * seq_len + t) * vocab + (size_t)id] = 1.0f;
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- GradCAM (Zhou-2016 closed form) ----
// For a network ending in (globalAvgPool → dense), the GradCAM
// heatmap collapses to:
//     heat[i, j] = ReLU( Σ_k denseW[k, tgt] · lastConv[k, i, j] )
// Up to a positive constant, that is exactly Zhou et al. 2016 CAM.
// Inputs:
//     dense_w   : f32 [C, NC] row-major  (final dense weights)
//     last_conv : f32 [B, C, H, W] NCHW  (pre-GAP feature maps)
//     batch_idx : which batch element to extract (typically 0)
//     C, H, W, NC, tgt
// Output: f32 [H, W] heatmap, ReLU-clamped, max-normalized to [0,1].
// (If max < 1e-6 we leave the buffer at 0 — a flat zero-attention map.)
LEAN_EXPORT lean_obj_res lean_f32_cam_compute(
    b_lean_obj_arg dense_w_ba, b_lean_obj_arg last_conv_ba,
    size_t batch_idx, size_t C, size_t H, size_t W, size_t NC,
    size_t tgt, lean_obj_arg w_io) {
    (void)w_io;
    const float* W_dense = (const float*)lean_sarray_cptr(dense_w_ba);
    const float* A = (const float*)lean_sarray_cptr(last_conv_ba);
    size_t plane = H * W;
    const float* A_b = A + batch_idx * C * plane;  // [C, H, W] for this image
    size_t nbytes = plane * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* heat = (float*)lean_sarray_cptr(out);
    // Σ_k W[k, tgt] · A[k, i, j], then ReLU.
    float mx = 0.0f;
    for (size_t i = 0; i < H; i++) {
        for (size_t j = 0; j < W; j++) {
            float s = 0.0f;
            for (size_t k = 0; k < C; k++) {
                s += W_dense[k * NC + tgt] * A_b[k * plane + i * W + j];
            }
            if (s < 0.0f) s = 0.0f;
            heat[i * W + j] = s;
            if (s > mx) mx = s;
        }
    }
    if (mx > 1e-6f) {
        float inv = 1.0f / mx;
        for (size_t k = 0; k < plane; k++) heat[k] *= inv;
    } else {
        for (size_t k = 0; k < plane; k++) heat[k] = 0.0f;
    }
    return lean_io_result_mk_ok(out);
}

// ---- Recompute logits from captured pre-GAP activation ----
// Used by the GradCAM exe when class selection is "auto" (argmax).
// logits[c] = Σ_k W[k, c] · gap_k + bias[c],  gap_k = (1/HW) Σ_ij A[k,i,j]
// Inputs:
//     dense_w   : f32 [C, NC] row-major
//     dense_b   : f32 [NC]
//     last_conv : f32 [B, C, H, W] NCHW
//     batch_idx : which batch element (typically 0)
// Output: f32 [NC] logits.
LEAN_EXPORT lean_obj_res lean_f32_cam_logits(
    b_lean_obj_arg dense_w_ba, b_lean_obj_arg dense_b_ba,
    b_lean_obj_arg last_conv_ba,
    size_t batch_idx, size_t C, size_t H, size_t W, size_t NC,
    lean_obj_arg w_io) {
    (void)w_io;
    const float* W_dense = (const float*)lean_sarray_cptr(dense_w_ba);
    const float* B_dense = (const float*)lean_sarray_cptr(dense_b_ba);
    const float* A = (const float*)lean_sarray_cptr(last_conv_ba);
    size_t plane = H * W;
    const float* A_b = A + batch_idx * C * plane;
    size_t nbytes = NC * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* logits = (float*)lean_sarray_cptr(out);
    // Compute global avg pool.
    float* gap = (float*)malloc(C * sizeof(float));
    float invHW = 1.0f / (float)plane;
    for (size_t k = 0; k < C; k++) {
        float s = 0.0f;
        for (size_t p = 0; p < plane; p++) s += A_b[k * plane + p];
        gap[k] = s * invHW;
    }
    for (size_t c = 0; c < NC; c++) {
        float s = B_dense[c];
        for (size_t k = 0; k < C; k++) s += W_dense[k * NC + c] * gap[k];
        logits[c] = s;
    }
    free(gap);
    return lean_io_result_mk_ok(out);
}

// ---- Bilinear upsample a single 2D plane (no channels) ----
// Input  : f32 [Hin, Win]   (e.g. 7x7 GradCAM heatmap)
// Output : f32 [Hout, Wout] (e.g. 224x224 to overlay on the image)
// Aligns corners (so first/last rows and cols map exactly), which is
// the standard Pillow / OpenCV behavior for visualization. Out-of-range
// indices are clamped to the edge.
LEAN_EXPORT lean_obj_res lean_f32_bilinear_upsample_2d(
    b_lean_obj_arg in_ba, size_t Hin, size_t Win, size_t Hout, size_t Wout,
    lean_obj_arg w_io) {
    (void)w_io;
    const float* in = (const float*)lean_sarray_cptr(in_ba);
    size_t nbytes = Hout * Wout * 4;
    lean_object* out_obj = lean_alloc_sarray(1, nbytes, nbytes);
    float* out = (float*)lean_sarray_cptr(out_obj);
    if (Hin == 0 || Win == 0) {
        memset(out, 0, nbytes);
        return lean_io_result_mk_ok(out_obj);
    }
    double sy = (Hout > 1) ? (double)(Hin - 1) / (double)(Hout - 1) : 0.0;
    double sx = (Wout > 1) ? (double)(Win - 1) / (double)(Wout - 1) : 0.0;
    for (size_t i = 0; i < Hout; i++) {
        double yy = (double)i * sy;
        size_t y0 = (size_t)yy;
        size_t y1 = y0 + 1; if (y1 >= Hin) y1 = Hin - 1;
        double dy = yy - (double)y0;
        for (size_t j = 0; j < Wout; j++) {
            double xx = (double)j * sx;
            size_t x0 = (size_t)xx;
            size_t x1 = x0 + 1; if (x1 >= Win) x1 = Win - 1;
            double dx = xx - (double)x0;
            float v00 = in[y0 * Win + x0];
            float v01 = in[y0 * Win + x1];
            float v10 = in[y1 * Win + x0];
            float v11 = in[y1 * Win + x1];
            double v0 = v00 + (v01 - v00) * dx;
            double v1 = v10 + (v11 - v10) * dx;
            out[i * Wout + j] = (float)(v0 + (v1 - v0) * dy);
        }
    }
    return lean_io_result_mk_ok(out_obj);
}

LEAN_EXPORT lean_obj_res lean_f32_rand_augment(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, size_t n_ops, double m,
    size_t imagenet_norm, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t plane = height * width;
    size_t pixels = channels * plane;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    memcpy(o, lean_sarray_cptr(images), nbytes);
    uint64_t s = seed ^ 0xd4e5f60718293a4bULL; if (s == 0) s = 1;
    const int N_KINDS = 5;  // identity, brightness, contrast, color, autocontrast
    for (size_t i = 0; i < batch; i++) {
        float* img = o + i * pixels;
        if (imagenet_norm) denormalize_imagenet(img, channels, plane);
        for (size_t k = 0; k < n_ops; k++) {
            int op = (int)(f32_xorshift64(&s) % N_KINDS);
            switch (op) {
                case 0: break;  // identity
                case 1: apply_brightness(img, pixels, rand_factor(m, &s)); break;
                case 2: apply_contrast(img, pixels, rand_factor(m, &s));   break;
                case 3: apply_color(img, channels, height, width, rand_factor(m, &s)); break;
                case 4: apply_autocontrast(img, pixels); break;
            }
            if (imagenet_norm) clamp01(img, pixels);
        }
        if (imagenet_norm) renormalize_imagenet(img, channels, plane);
    }
    return lean_io_result_mk_ok(out);
}
