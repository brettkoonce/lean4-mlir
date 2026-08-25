#!/usr/bin/env python3
"""Gate 2 for the bf16 render path: did the bf16 actually reach the hardware?

    .venv/bin/python scripts/bf16_gate2.py verified_mlir/resnet34in_momdp64bf16_train_step.mlir
    .venv/bin/python scripts/bf16_gate2.py <bf16.mlir> --against <f32.mlir>   # + a speedup

⚠⚠ WHY THIS EXISTS, AND WHY IT DOES NOT GREP. A `stablehlo.convolution` line carries only its
RESULT type. Grepping it for "bf16" reports success for a graph whose operands XLA has quietly
converted back to f32 — which is exactly what happens to a conv with bf16 operands and an
f32-typed result (`xla_allow_excess_precision`, and `=false` does not rescue it). That mistake
was made twice in one session before this script existed. So: compile, then resolve each
convolution's OPERAND SSA names to their dtypes in the OPTIMIZED HLO.

▶ `planning/bf16_renderer.md` §9.2 has the measurement; `BatchableOp.convBf16` has the note.
"""
import argparse, re, sys, time

def _lazy():
    import jax, jax.extend as jex, numpy as np
    from jax._src.lib import xla_client as xc
    from jax._src.interpreters import mlir as jmlir
    from jax._src import compiler as jcomp
    from jaxlib.mlir import ir
    return jax, jex, np, xc, jmlir, jcomp, ir

def compile_mlir(path, ctxs):
    jax, jex, np, xc, jmlir, jcomp, ir = ctxs
    client = jex.backend.get_backend()
    dl = xc.DeviceList(tuple(client.local_devices()[:1]))
    src = open(path).read()
    # XLA insists the entry be `main`; the artifacts name theirs after the variant
    # (`@resnet34in_momdp64bf16_train_step`) because the driver matches on it. Rename for the
    # check only — this rewrites nothing on disk and changes no op in the body.
    src, n = re.subn(r"func\.func\s+(public\s+)?@[\w.]+\(", "func.func public @main(", src, count=1)
    if n == 0:
        raise SystemExit(f"{path}: no `func.func @…(` found — not a rendered artifact?")
    ctx = jmlir.make_ir_context()
    with ctx, ir.Location.unknown(ctx):
        mod = ir.Module.parse(src)
        return jcomp.backend_compile_and_load(client, mod, dl, xc.CompileOptions(), [])

def conv_operand_dtypes(exe):
    """[(op-kind, [operand dtypes])] for every convolution in the optimized HLO."""
    txt = exe.hlo_modules()[0].to_string()
    types = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(%[\w.\-]+) = ([a-z0-9]+)\[", line.strip())
        if m:
            types[m.group(1)] = m.group(2)
    out = []
    for line in txt.splitlines():
        s = line.strip()
        is_conv = ("cudnn" in s and "custom-call" in s) or re.search(r"=\s*\S+\s+convolution\(", s)
        if not is_conv:
            continue
        g = re.search(r"(?:custom-call|convolution)\(([^)]*)\)", s)
        if g:
            names = [o.strip() for o in g.group(1).split(",")][:2]
            out.append((("cudnn" if "cudnn" in s else "hlo"), [types.get(n, "?") for n in names]))
    return out

def _computations(txt):
    """[(header, [body lines])] — one entry per HLO computation, entry included.

    Needed because XLA puts a matmul-heavy net's dots inside `gemm_fusion_dot.*_computation`
    fusions, where the operands are that computation's PARAMETERS and their dtypes are declared in
    its header. A single global name->dtype map (which the convolution path above can afford, since
    cuDNN custom-calls stay in the entry) would miss every one of them.
    """
    out, cur, body = [], None, []
    for line in txt.splitlines():
        if re.match(r"^\s*(ENTRY\s+)?[%\w.\-]+\s*\(.*\{\s*$", line) and " = " not in line:
            if cur is not None:
                out.append((cur, body))
            cur, body = line.strip(), []
        elif cur is not None:
            body.append(line)
    if cur is not None:
        out.append((cur, body))
    return out

def dot_operand_dtypes(exe):
    """[(kind, [operand dtypes])] for every DOT in the optimized HLO, fused ones included.

    ⚠⚠ SAME REASON THIS FILE DOES NOT GREP. A `dot` line carries only its RESULT type, and for a
    `dot_general` the result type is f32 BY DESIGN in this repo's emit shape (bf16 operands, f32
    accumulate — `planning/bf16_renderer.md` §9.2). So on a matmul-bound net, grepping result types
    reports "no bf16 anywhere" for a perfectly good bf16 graph, which is the mirror image of the
    convolution mistake above. Only the OPERANDS answer the question.
    """
    txt = exe.hlo_modules()[0].to_string()
    out = []
    for header, body in _computations(txt):
        types = {}
        # the computation's own parameters, declared in its header as `name: dtype[dims]`
        sig = re.search(r"\((.*)\)", header)
        if sig:
            for nm, dt in re.findall(r"([%\w.\-]+):\s*([a-z0-9]+)\[", sig.group(1)):
                types[nm] = dt
                types["%" + nm.lstrip("%")] = dt
        for line in body:
            m = re.match(r"\s*(?:ROOT\s+)?(%[\w.\-]+) = ([a-z0-9]+)\[", line.strip())
            if m:
                types[m.group(1)] = m.group(2)
        for line in body:
            s = line.strip()
            is_gemm = "__cublas" in s and "custom-call" in s
            if not (is_gemm or re.search(r"=\s*\S+\s+dot\(", s)):
                continue
            g = re.search(r"(?:custom-call|dot)\(([^)]*)\)", s)
            if g:
                names = [o.strip() for o in g.group(1).split(",")][:2]
                out.append((("cublas" if is_gemm else "dot"), [types.get(n, "?") for n in names]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--against", help="the f32 peer, to also report a speedup")
    ap.add_argument("--expect", default="bf16", choices=["bf16", "f32"],
                    help="dtype every convolution operand must have (default bf16)")
    ap.add_argument("--dot-results", action="store_true",
                    help="also report each bf16 dot's RESULT dtype. Not a correctness check — bf16 "
                         "reaches the hardware either way — but an f32 result costs ~1.2x on a real "
                         "chain (bf16_renderer.md §20.1), so a bf16 dot with an f32 result is a "
                         "PERFORMANCE defect worth seeing. Some are deliberate: ViT's weight "
                         "gradients keep f32 results because their output is a small weight.")
    ap.add_argument("--dots", action="store_true",
                    help="ALSO resolve every dot/gemm's operand dtypes and print the breakdown. "
                         "Required for matmul-bound nets (ViT): their convolution count is 2 and a "
                         "conv-only gate would report green while saying nothing about the 90%% of "
                         "the step that is dots. Prints counts rather than asserting, because a net "
                         "legitimately keeps some dots in f32 (every classifier head here does).")
    a = ap.parse_args()
    ctxs = _lazy()

    exe = compile_mlir(a.mlir, ctxs)
    convs = conv_operand_dtypes(exe)
    if not convs:
        print(f"⛔ no convolution found in {a.mlir} — is this the right artifact?")
        return 1
    bad = [(k, d) for k, d in convs if not all(x == a.expect for x in d)]
    print(f"  {a.mlir}")
    print(f"  {len(convs)} convolutions; {len(convs) - len(bad)} with all operands {a.expect}")
    if bad:
        print(f"\n  ⛔ GATE 2 FAILED — {len(bad)} convolution(s) not {a.expect}:")
        for k, d in bad[:8]:
            print(f"       {k}: operands={d}")
        print("\n  ▶ If these are f32 in a bf16 artifact, the emit gave the convolution an")
        print("    f32-TYPED result and XLA deleted the casts. See bf16_renderer.md §9.2.")
        return 1
    what = ("bf16 reached the hardware in every convolution" if a.expect == "bf16"
            else "every convolution is f32, as expected of the control arm")
    print(f"  ✅ GATE 2: {what}")

    if a.dots:
        dots = dot_operand_dtypes(exe)
        tally = {}
        for k, d in dots:
            key = "all bf16" if all(x == "bf16" for x in d) else (
                  "all f32" if all(x == "f32" for x in d) else "MIXED " + "/".join(d))
            tally[key] = tally.get(key, 0) + 1
        print(f"\n  {len(dots)} dot/gemm instructions (fused ones included):")
        for k in sorted(tally, key=lambda k: -tally[k]):
            print(f"       {tally[k]:5d}  operands {k}")
        print("  ▶ These are the OPERANDS, resolved per computation. A dot's RESULT type is a")
        print("    SEPARATE question and it is not a correctness one: bf16 reaches the tensor")
        print("    cores either way (§9.2), but an f32 result makes the gemm write twice the")
        print("    bytes and cost ~1.2x on a real chain (§20.1). Check it with --dot-results.")

    if a.dot_results:
        txt = exe.hlo_modules()[0].to_string()
        nb = len(re.findall(r"=\s*bf16\[[^\]]*\][^=]*?\bdot\(", txt))
        nf = len(re.findall(r"=\s*f32\[[^\]]*\][^=]*?\bdot\(", txt))
        print(f"\n  dot RESULT dtypes: {nb} bf16, {nf} f32")
        if nf:
            print(f"  ⚠ {nf} dot(s) accumulate straight into f32. That is correct and it is not")
            print(f"    free — see §20.1. Deliberate for small weight-gradient results; a defect")
            print(f"    for anything producing a large activation.")

    if a.against:
        jax, jex, np, *_ = ctxs
        ref = compile_mlir(a.against, ctxs)
        print(f"\n  ⚠ speedup not measured: it needs matching input buffers for both modules,")
        print(f"    which this script does not synthesize. Use LEAN_MLIR_MAX_STEPS on the real")
        print(f"    trainer instead — that is the number worth quoting anyway (§10.4 step 6).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
