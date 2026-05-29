# HAL cloud-training image (multi-stage). The builder compiles the venv —
# Rust + headers for the peppi-py/libmelee git forks; the runtime keeps only the
# built /opt/venv + interpreter, dropping the toolchain. torch bundles its own
# CUDA libs, so the runtime needs no compiler or CUDA devel toolkit.
#
# The image carries the *environment*, never the code: the venv holds the deps
# (incl. the compiled forks) but not the `hal` package. Code is layered in at
# runtime — `docker/on-start.sh` clones the target git SHA on vast; compose
# bind-mounts the working tree locally — and a runtime `uv sync` installs the
# pure-Python `hal` into the prebuilt venv (seconds, no compiler). So editing
# code never invalidates this image; only a uv.lock change forces a rebuild.
#
# Base recommended by vast.ai: it ships uv + the UV_* env (UV_PROJECT_ENVIRONMENT
# =/opt/venv, UV_PYTHON_INSTALL_DIR=/.uv/python_install, UV_NO_CACHE=1) and the
# SSH/Jupyter provisioning that vast's launch modes depend on — so both stages
# use it rather than a leaner-but-incompatible base.

# ---- builder: resolve + compile the venv ----
FROM vastai/base-image:cuda-13.0.2-auto AS builder

# The base sets UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR / UV_LINK_MODE / UV_NO_CACHE
# but NOT these, so set them here: the venv must land at /opt/venv (default is
# .venv in the workdir) on the managed CPython, with bytecode precompiled.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    PATH=/root/.cargo/bin:$PATH

# Build-only deps: compilers + headers (the -dev packages). Runtime libs come
# from the base / the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config \
      libasound2-dev libegl-dev libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (peppi-py builds via maturin).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
      --default-toolchain stable --profile minimal

WORKDIR /opt/hal

# Deps only — no project. Downloads managed CPython 3.14, every pinned wheel, and
# compiles the git forks into /opt/venv. `hal` itself is deliberately NOT installed
# (no `COPY . .`): it's pure-Python and gets installed at runtime from the
# cloned/mounted code, so this layer caches across all code edits — only uv.lock
# busts it.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# ---- runtime: slim; just the venv + interpreter + runtime libs ----
FROM vastai/base-image:cuda-13.0.2-auto

# UV_PROJECT_ENVIRONMENT so `uv run` resolves the copied venv instead of making
# a fresh .venv; only-managed pins it to the copied interpreter under /.uv.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH=/opt/venv/bin:$PATH

# Runtime-only system deps (no compilers, no -dev): xvfb gives Dolphin a virtual
# display for the closed-loop eval; p7zip extracts fixture archives; the GTK/GL
# libs back libmelee's Dolphin.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git p7zip-full xvfb \
      libasound2t64 libegl1 libgl1 libusb-1.0-0 \
      libglib2.0-0t64 libgdk-pixbuf-2.0-0 libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# /opt/hal is left empty: code is cloned (vast) or bind-mounted (compose) at runtime.
WORKDIR /opt/hal

# The venv references the managed interpreter by absolute path, so copy both. No
# code copied — only the environment.
COPY --from=builder /.uv/python_install /.uv/python_install
COPY --from=builder /opt/venv /opt/venv

# Both launch scripts live outside /opt/hal so the runtime clone into /opt/hal
# can't clobber the running script. entrypoint.sh backs local `docker run`/compose;
# on-start.sh is the vast cloud lifecycle (clone SHA -> train -> self-teardown).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/on-start.sh /usr/local/bin/on-start.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/on-start.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
