#!/usr/bin/env python3
"""Write <run>/manifest.json for a platform-suite run (planning/platform_integration.md §4).

Two halves, so a red run can be diffed against the last green one:

  core      what WE shipped: repo SHA, shim source + binary hashes, compile-options header,
            Lean toolchain, the verified_mlir manifest, the suite's own sources.
  platform  what the BOX supplied: GPUs, kernel driver, CUDA/ROCm runtime, PJRT plugin
            (path, package version, hash, API version), kernel, compiler.

Core equal and platform changed means the driver or plugin; platform equal and core changed
means us. Called by scripts/platform/check.sh; reads the probe log it left in <run>/logs.
"""
import argparse, hashlib, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None


def sh(*cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def tree_sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def plugin_package(plugin):
    """The pip distribution that ships the plugin: the *pjrt* / *plugin* dist-info beside it."""
    sp = next((p for p in Path(plugin).resolve().parents if p.name == 'site-packages'), None)
    if not sp: return None
    for pat in ('*pjrt*.dist-info', '*plugin*.dist-info'):
        for d in sorted(sp.glob(pat)):
            m = re.match(r'(.+)-([^-]+)\.dist-info$', d.name)
            if m: return f'{m.group(1)}=={m.group(2)}'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--backend', required=True)
    ap.add_argument('--plugin', default=''); ap.add_argument('--tier', type=int, default=1)
    a = ap.parse_args()
    run = Path(a.run)
    probe_log = run / 'logs' / 'probe.log'
    probe = {}
    xla_line = ''
    if probe_log.exists():
        for ln in probe_log.read_text(errors='replace').splitlines():
            m = re.match(r'^([a-z_0-9]+)=(.*)$', ln)
            if m: probe[m.group(1)] = m.group(2)
            if 'StreamExecutor [0]' in ln: xla_line = ln.split(': ', 1)[-1]
    xla = dict(re.findall(r'(Driver|Runtime|Toolkit|DNN): ([^;)\]]+[\]]?)', xla_line))

    dirty = bool(sh('git', 'status', '--porcelain', '--untracked-files=no', '--', '.', ':!runs'))
    core = {
        'repo_sha': sh('git', 'rev-parse', 'HEAD'),
        'repo_dirty': dirty,
        'lean_toolchain': (ROOT / 'lean-toolchain').read_text().strip(),
        'shim_source_sha': sha(ROOT / 'ffi/pjrt_ffi.c'),
        'shim_binary_sha': sha(run / 'build/libpjrt_ffi.so'),
        'compile_options_sha': sha(ROOT / 'ffi/pjrt_compile_options.h'),
        'pjrt_api_header': probe.get('pjrt_api_header'),
        'verified_mlir_manifest_sha': sha(ROOT / 'verified_mlir/MANIFEST.md'),
        'suite_sha': tree_sha([p for p in (ROOT / 'scripts/platform').rglob('*')
                               if p.is_file() and p.parent.name != 'expected'
                               and '__pycache__' not in p.parts]),
        'tier': a.tier,
    }
    gpus = []
    if a.backend == 'cuda':
        vis = os.environ.get('CUDA_VISIBLE_DEVICES')
        q = ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader']
        if vis: q += ['-i', vis]
        for ln in sh(*q).splitlines():
            f = [x.strip() for x in ln.split(',')]
            if len(f) != 4 or not f[0].isdigit():   # a Jetson's nvidia-smi may refuse a field
                continue
            idx, name, mem, drv = f
            gpus.append({'index': int(idx), 'name': name, 'memory': mem, 'driver': drv})
    if not gpus:   # ROCm / XPU: the plugin's own device kinds, until their SMIs are parsed
        gpus = [{'index': int(k.split('_')[1]), 'name': v, 'memory': None, 'driver': None}
                for k, v in sorted(probe.items()) if re.match(r'device_\d+$', k)]
    plat = {
        'backend': a.backend,
        'host': platform.node().split('.')[0],
        'gpus': gpus,
        'gpu_count_visible': int(probe.get('devices', 0) or 0),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('HIP_VISIBLE_DEVICES'),
        'driver': gpus[0]['driver'] if gpus else None,
        'runtime': {k.lower(): v.strip() for k, v in xla.items()},
        'plugin_path': a.plugin,
        'plugin_package': plugin_package(a.plugin) if a.plugin else None,
        'plugin_sha': sha(ROOT / a.plugin if a.plugin and not os.path.isabs(a.plugin) else a.plugin),
        'pjrt_api_plugin': probe.get('pjrt_api_plugin'),
        'platform_name': probe.get('platform_name'),
        'platform_version': probe.get('platform_version'),
        'kernel': platform.release(),
        'cc': sh(os.environ.get('CC', 'gcc'), '--version').splitlines()[0] if sh('gcc', '--version') else None,
    }
    m = {'date': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
         'run': run.name, 'core': core, 'platform': plat}
    (run / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
