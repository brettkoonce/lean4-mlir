#include <lean/lean.h>
#include <stdio.h>
#include <errno.h>

/* Read up to `len` bytes from `h` into `buf`, APPENDING at buf.size and never touching
   capacity. `buf` must be exclusive (RC 1) with capacity >= size + len. Returns the buffer
   with its size advanced by the bytes read (0 at EOF, like Handle.read). */
LEAN_EXPORT lean_obj_res lean_mlir_read_into(b_lean_obj_arg h, lean_obj_arg buf, size_t len, lean_obj_arg w) {
    (void)w;
    /* Exclusive means ONE reference. Lean's own lean_is_exclusive says false for any
       multi-threaded object (an object that has crossed a Task boundary keeps a negated,
       atomic RC for life), so it is checked by hand: st rc == 1, or mt rc == -1. */
    int rc = buf->m_rc;
    if (!(rc == 1 || rc == -1)) {
        char msg[128];
        snprintf(msg, sizeof msg, "read_into: buffer is shared (rc=%d, %s), refusing to mutate in place",
                 rc, lean_is_st(buf) ? "st" : "mt");
        lean_dec(buf);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg)));
    }
    size_t sz = lean_sarray_size(buf), cap = lean_sarray_capacity(buf);
    if (sz + len > cap) {
        lean_dec(buf);
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("read_into: size + len exceeds the buffer's capacity")));
    }
    FILE * fp = (FILE *)lean_get_external_data(h);
    size_t n = fread(lean_sarray_cptr(buf) + sz, 1, len, fp);
    if (n == 0 && len > 0 && ferror(fp)) {
        int e = errno; clearerr(fp); lean_dec(buf);
        return lean_io_result_mk_error(lean_decode_io_error(e, NULL));
    }
    if (feof(fp)) clearerr(fp);
    /* not lean_sarray_set_size: its debug assert is lean_is_exclusive, which is false for
       every mt object. The rc check above is the real exclusivity condition. */
    lean_to_sarray(buf)->m_size = sz + n;
    return lean_io_result_mk_ok(buf);
}
