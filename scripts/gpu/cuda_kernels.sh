# shellcheck shell=bash
#
# Sourced by the CUDA build script and by every GPU job script, after the CUDA module is
# loaded and a Go new enough for go.mod is on PATH:
#
#   . scripts/gpu/cuda_kernels.sh && adcgo_cuda_kernels || exit 1
#
# After it, `go build -tags cuda` and `go test -tags cuda` link in this shell. A cuda build
# links four kernel objects that nvcc has to compile first, and Go does not do it:
#
#   ADCgo's own   internal/adc/devkern/{adc4_kernels,adc2dip_kernels}.o
#                 (the CVS-ADC(4) wert2 and order-3 SIP c22 appliers; the DIP satellite)
#   goHPC's       backend/{tensor_kernels,bandeig_kernels}.o
#                 (the generated sigma-build; the banded eigensolver's replay)
#
# ADCgo's are compiled beside their sources when missing or older than them. goHPC is a
# tagged module in the read-only module cache, so it is copied once per version into
# .build/goHPC-<version>, compiled there by its own scripts/build_cuda_kernels, and the
# build is pointed at the copy by a workspace file: GOWORK is exported, go.mod is not
# touched. A goHPC checkout that a go.work of your own already names is compiled in place.
#
# Every object is built for sm_80 (A100), sm_86 (A40) and sm_90 (H200) and never for
# fewer: the objects are shared by every job in this checkout, and one compiled for a
# single model fails at run time on another.
#
# Two jobs may get here at once. Each object, the goHPC copy and the workspace file are
# written under a job-unique name and renamed into place; rename(2) is atomic, so a
# concurrent build links either the old object or the new one, never half of one.
#
# NVCC_EXTRA      extra nvcc flags for ADCgo's kernels (-lineinfo, "-Xptxas -v")
# KERNELS_FORCE   1 recompiles ADCgo's kernels although they are current
#
# Exports GOWORK (when it made the copy) and GOHPC_DIR, the directory of the goHPC the
# build uses -- its backend tests are `go test github.com/leiaSQ/goHPC/backend` and run
# from $GOHPC_DIR/backend.

adcgo_cuda_kernels() {
    local root id nvcc gencode arch ker cu obj need gohpc ver copy tmp
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
    id="${SLURM_JOB_ID:-$$}"

    nvcc="$(command -v nvcc 2>/dev/null || true)"
    [ -n "$nvcc" ] || nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
    [ -x "$nvcc" ] || { echo "cuda_kernels: nvcc not found; module load devel/cuda/13.2" >&2; return 1; }
    command -v go >/dev/null 2>&1 || { echo "cuda_kernels: go not found" >&2; return 1; }

    gencode=""
    for arch in 80 86 90; do
        gencode="$gencode -gencode arch=compute_$arch,code=sm_$arch"
    done

    for ker in adc4_kernels adc2dip_kernels; do
        cu="$root/internal/adc/devkern/$ker.cu"
        obj="$root/internal/adc/devkern/$ker.o"
        if [ "${KERNELS_FORCE:-0}" = 1 ] || [ ! -f "$obj" ] || [ "$cu" -nt "$obj" ]; then
            echo "nvcc: $ker.cu"
            # shellcheck disable=SC2086
            "$nvcc" -O3 -std=c++14 ${NVCC_EXTRA:-} $gencode -c "$cu" -o "$obj.$id" || {
                rm -f "$obj.$id"
                echo "cuda_kernels: nvcc failed for $ker.cu" >&2
                return 1
            }
            mv -f "$obj.$id" "$obj" || return 1
        else
            echo "nvcc: $ker.o is up to date"
        fi
    done

    gohpc="$(cd "$root" && go list -m -f '{{.Dir}}' github.com/leiaSQ/goHPC)" || return 1
    [ -n "$gohpc" ] && [ -d "$gohpc/backend" ] || {
        echo "cuda_kernels: could not locate github.com/leiaSQ/goHPC (got '$gohpc')" >&2
        return 1
    }
    if [ -w "$gohpc/backend" ]; then
        "$gohpc/scripts/build_cuda_kernels" >/dev/null || return 1
    else
        ver="$(cd "$root" && go list -m -f '{{.Version}}' github.com/leiaSQ/goHPC)" || return 1
        copy="$root/.build/goHPC-$ver"
        if [ ! -f "$copy/.built" ]; then
            mkdir -p "$root/.build" || return 1
            tmp="$copy.tmp.$id"
            rm -rf "$tmp"
            bash "$gohpc/scripts/build_cuda_kernels" "$tmp" >/dev/null || {
                rm -rf "$tmp"
                echo "cuda_kernels: compiling goHPC $ver's kernels failed" >&2
                return 1
            }
            touch "$tmp/.built"
            # Lost the race to another job: its copy is the same code, keep it.
            mv -T "$tmp" "$copy" 2>/dev/null || rm -rf "$tmp"
            [ -f "$copy/.built" ] || { echo "cuda_kernels: no goHPC copy at $copy" >&2; return 1; }
        fi
        need="$(awk '/^go /{print $2; exit}' "$root/go.mod")"
        printf 'go %s\n\nuse (\n\t%s\n\t%s\n)\n' "$need" "$root" "$copy" > "$root/.build/go.work.$id" || return 1
        mv -f "$root/.build/go.work.$id" "$root/.build/go.work" || return 1
        GOWORK="$root/.build/go.work"
        export GOWORK
        gohpc="$copy"
    fi
    GOHPC_DIR="$gohpc"
    export GOHPC_DIR
    echo "goHPC     = $GOHPC_DIR"
    [ -n "${GOWORK:-}" ] && echo "GOWORK    = $GOWORK"
    return 0
}
