# Lc0: build and install

Leela Chess Zero v0.32.1 (tag `v0.32.1`, commit `fd71a2d921b689c5f479d3227c3806c8e272d9c5`,
released 2025-11-23), built from source without root, with the CUDA backend and the
Python bindings. `tnh.models.lc0` finds it through `TNH_LC0` if set, else
`~/.local/opt/lc0/`.

What ends up installed:

```
~/.local/opt/lc0/bin/lc0
~/.local/opt/lc0/lib/python3.12/site-packages/lczero/backends.cpython-312-x86_64-linux-gnu.so
~/.local/opt/lc0/networks/BT4-1024x15x32h-swa-6147500.pb.gz
~/.local/bin/lc0        wrapper script: starts lc0 with the network above
```

## Requirements

Checked on the G14 (Ubuntu 26.04, RTX 4070 Laptop, driver 595.99 with CUDA 13.2):

- `nvcc` from the apt package `nvidia-cuda-toolkit` (12.4) and its cuBLAS headers and
  libraries. A driver that supports a newer CUDA runs 12.4 code.
- A host compiler nvcc 12.4 accepts: it rejects gcc 15, so `g++-13` is given to nvcc
  with `-Dnvcc_ccbin=g++-13`. The rest of lc0 builds with the system compiler.
- `ninja`, `git`, `zlib1g-dev`, `pkg-config`.
- `meson`, run through `uv tool run` (no install needed).
- The Python the bindings are for, with headers. The project uses uv's CPython 3.12;
  its `lib/pkgconfig/python3.pc` is found through `PKG_CONFIG_PATH`. The bindings only
  load in the same Python minor version.

No cuDNN: the `plain_cuda` backend uses cuBLAS and lc0's own kernels.

## Build

```bash
B=~/out/tn-hittare/tools/lc0-build        # disposable
PY=$(dirname $(dirname $(readlink -f ~/projs/tn-hittare/.venv/bin/python)))
mkdir -p $B && cd $B
git clone --depth 1 --branch v0.32.1 https://github.com/LeelaChessZero/lc0.git src
export PATH=$HOME/projs/tn-hittare/.venv/bin:$PATH PKG_CONFIG_PATH=$PY/lib/pkgconfig
uv tool run --from meson==1.12.1 meson setup build src --buildtype=release \
    --prefix=$HOME/.local/opt/lc0 -Dnvcc_ccbin=g++-13 -Dplain_cuda=true -Dcudnn=false \
    -Dpython_bindings=true -Donnx=false -Dgtest=false
ninja -C build -j 8                       # under half an hour on the G14 (not timed)
uv tool run --from meson==1.12.1 meson install -C build
rm -rf ~/.local/opt/lc0/include ~/.local/opt/lc0/lib/x86_64-linux-gnu   # Eigen headers from a subproject
```

Check that meson's summary says `Program python3 found: YES` with Python 3.12, and
`Library cublas found: YES`. The build uses `-march=native` and CUDA code for the
local GPU only, so build again on another machine rather than copying the binary.
(`meson install --skip-subprojects` should leave the Eigen headers out; not tried.)

## Network

BT4-1024x15x32h-swa-6147500, the base BT4 network (not the policy-tuned `it332` that
is sent to engine competitions), from the official Lc0 storage:

```bash
mkdir -p ~/.local/opt/lc0/networks && cd ~/.local/opt/lc0/networks
curl -O https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz
sha256sum BT4-1024x15x32h-swa-6147500.pb.gz
# 66e6c1b5308c34230801dc171df580d7a66ec916dc2211246a485ef8e545c67e  (335,897,546 bytes,
# Last-Modified 2024-02-22, downloaded 2026-11-11)
```

## On the PATH

`~/.local/bin/lc0` is a wrapper, because lc0's network autodiscovery only looks in
`./` and `./weights/`:

```sh
#!/bin/sh
LC0_HOME="${TNH_LC0:-$HOME/.local/opt/lc0}"
case "$1" in
  ""|-*) exec "$LC0_HOME/bin/lc0" --weights="$LC0_HOME/networks/BT4-1024x15x32h-swa-6147500.pb.gz" "$@" ;;
  *) exec "$LC0_HOME/bin/lc0" "$@" ;;   # a mode such as benchmark or leela2onnx
esac
```

Test from any directory:

```bash
cd /tmp && printf 'uci\nisready\nposition startpos\ngo nodes 1\nquit\n' | lc0
```

## Settings that matter, and why

- **Batches of at most 12.** In v0.32.1 the CUDA backend gives non-deterministic
  results with BT4 for batches of 16 and more: the same batch evaluated twice differs
  by up to 0.2 in a policy logit in float16 and up to 5 in float32. Batches up to 12
  repeat exactly, and in float32 equal batch 1 exactly. `tnh.models.lc0` therefore
  uses `max_batch=12` in the bindings and `BackendOptions max_batch=12`,
  `MinibatchSize 12`, `MaxPrefetch 0` in the engine (prefetching adds positions to a
  batch beyond the minibatch, and over `max_batch` the backend fails with
  `CUDA error: invalid argument`). Measured with
  `~/out/tn-hittare/scratch/models/scripts/lc0_stable.py`. Throughput costs little:
  1,714 positions/s at 12 against 1,921 at 32.
- **GPU memory.** With the default `max_batch` (1024) lc0 ran out of memory next to
  other models on the 8 GB card. With 12 an engine process takes about 1 GB, the
  bindings about the same.
- **Determinism of a search.** `Threads 1`, `TaskWorkers 0`, `NNCacheSize 0` and
  `ucinewgame` before each position make a search repeat exactly, in the same process
  and across processes. With the default task workers the tree differed between runs.
- **Exact values.** The verbose move stats print Q, WL and D after a round trip through
  lc0's approximate `FastLog`/`FastLogistic` (off by up to 0.003) and P through
  `FastExp`. `ScoreType Q` with `MultiPV 500` gives every root move's Q as
  `score cp` = 10000 Q, unapproximated.

## The Python bindings and canonical networks

`GameState.policy_indices()` maps moves to policy indices without the board transform
that `GameState.as_input()` applies for canonical-input networks (input format 3 and
above: a horizontal flip when castling rights are gone and the king is on files a-d,
and more in pawnless positions). For such a network the bindings, used as documented,
return the policy of the wrong moves in those positions. BT4-1024x15x32h-swa-6147500
has the classical input format (1), so no transform applies and the bindings are right
for it; this was confirmed against the engine. `tnh.models.lc0` ports the transform for
canonical networks (T3, T1 and other distilled nets), but that path has not been
checked against the engine.
