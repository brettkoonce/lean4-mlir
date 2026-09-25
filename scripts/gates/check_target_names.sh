#!/usr/bin/env bash
# check_target_names.sh — every binary a RUNNABLE file names must be a real lake target.
#
#   scripts/gates/check_target_names.sh          # lint; exit 1 on a dead name
#
# ▶ WHY THIS EXISTS. `lake build` builds `Proofs` and nothing else, and until
# 2026-09-20 no workflow built the apps/ exes, so a lean_exe could be renamed or
# deleted and every script driving it kept "working" — on the box where it was
# written, because the pre-rename binary was still sitting in .lake/build/bin.
# Measured 2026-09-20: `scripts/gates/residency_gate_all.sh` had carried
# `mnist-cnn-verified-xla`, `cifar8-bn-verified-xla`, `cifar8-bn-verified-adam-xla`
# and `resnet34-verified-adam-xla` since the 2026-08-10 rename — four of its eleven
# rows — reporting SKIP for each on a clean tree while still exiting 0. Six weeks,
# green the whole time.
#
# Deliberately a NAME check and not a build: pure grep, under a second, no
# toolchain, so it sits on every push. The build-level companion is
# `lake build Apps` (every exe entry point type-checked, no linking) in certs.yml.
#
# ▶ WHAT IT SCANS. Executable drivers only — scripts/*.sh, scripts/*/*.sh, tests/*.sh, run.sh and
# .github/workflows/*.yml. Comment lines are skipped, because half the prose in
# this repo explains what a lake subcommand does. Documentation is not scanned at
# all: historical/, the book and the run logs cite binaries by the name they had
# when the number was produced, and rewriting those would falsify the record.
#
# ▶ REFERENCE FORMS. `.lake/build/bin/<name>`, `lake {exe,run,build} <name>`, and a
# marker (see MARKER below) for table-driven drivers whose binaries sit inside a
# `|`-separated array no generic pattern can see. Tokens holding a shell variable,
# a glob, a brace or a dot (module paths) are skipped rather than guessed at.
#
# ▶ PACKAGES. jax/ is its own Lake package with its own targets, and jax.yml sets
# `working-directory: jax` per STEP rather than per job, so that one file is linted
# against BOTH lakefiles: a name live in either package is not a dead name, which is
# all this gate claims. comparator.yml builds two repos that are not in this tree at
# all (~/lean4export, ~/comparator); those two names are allowed below.
set -uo pipefail
cd "$(dirname "$0")/../.."

MARKER='lint-targets'            # a driver declares extra names with `# '"$MARKER"': a b c`
SELF=scripts/gates/check_target_names.sh

targets_of() {  # $1 = lakefile path; prints every name `lake` answers to there
  { grep -oP '^lean_exe\s+«\K[^»]+'  "$1"
    grep -oP '^lean_lib\s+«\K[^»]+'  "$1"
    grep -oP '^script\s+«\K[^»]+'    "$1"
    grep -oP '^script\s+\K[a-zA-Z][a-zA-Z0-9_-]*' "$1"; } 2>/dev/null | sort -u
}
mapfile -t ROOT_TARGETS < <(targets_of lakefile.lean; printf 'cache\n')
mapfile -t JAX_TARGETS  < <(targets_of jax/lakefile.lean; printf 'cache\n')
# Not targets of any lakefile in this tree — external checkouts comparator.yml clones.
EXTERNAL="lean4export comparator"

BAD=0; CHECKED=0
FILES=$(ls scripts/*.sh scripts/*/*.sh tests/*.sh run.sh .github/workflows/*.yml 2>/dev/null | grep -v "^$SELF\$")

for f in $FILES; do
  if [ "$f" = ".github/workflows/jax.yml" ]
  then set -- "${ROOT_TARGETS[@]}" "${JAX_TARGETS[@]}"
  else set -- "${ROOT_TARGETS[@]}"; fi
  ok=" $* $EXTERNAL "
  while IFS= read -r line; do
    ln=${line%%:*}; rest=${line#*:}
    # The marker is ITSELF a comment, so read it before the comment filter runs.
    names=$(printf '%s\n' "$rest" | grep -oP "#\s*$MARKER:\s*\K.*")
    if [ -z "$names" ]; then
      printf '%s' "$rest" | grep -qP '^\s*#' && continue        # a comment, not a command
      names=$(printf '%s\n' "$rest" | grep -oP '(?:\.lake/build/bin/|lake\s+(?:exe|run|build)\s+)\K[A-Za-z0-9._-]+')
    fi
    for n in $names; do
      case "$n" in *'$'*|*'*'*|*'{'*|*.*) continue;; esac
      CHECKED=$((CHECKED+1))
      case "$ok" in *" $n "*) ;; *) printf "  ✗ %s:%s  «%s» is not a lake target\n" "$f" "$ln" "$n"; BAD=1;; esac
    done
  done < <(grep -nP "\.lake/build/bin/|lake\s+(exe|run|build)\s|#\s*$MARKER:" "$f" 2>/dev/null)
done

if [ $BAD -eq 0 ]; then
  echo "✓ target-name lint: $CHECKED references across $(echo "$FILES" | wc -w) files, every one a live lake target"
  exit 0
fi
echo
echo "✗ dead target names above. Either the target was renamed (fix the caller) or"
echo "  deleted (delete the caller — or the gate it feeds is silently not running)."
exit 1
