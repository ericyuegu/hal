# HAL cloud-training image. Base recommended by vast.ai; we layer the repo's
# apt deps + uv-managed Python 3.14 + uv.lock-pinned wheels (incl. the libmelee
# and peppi-py git forks, the latter compiled via Rust/maturin).
FROM vastai/base-image:cuda-13.0.2-auto

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:$PATH

# Runtime + build deps. The first block mirrors scripts/setup.sh (validated on
# Linux); xvfb gives Dolphin a virtual display for the closed-loop eval.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config p7zip-full xvfb \
      libasound2t64 libasound2-dev libegl1 libegl-dev libgl1 \
      libusb-1.0-0 libusb-1.0-0-dev libglib2.0-0t64 libgdk-pixbuf-2.0-0 libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv (Python/env manager) + Rust toolchain (peppi-py builds via maturin).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
      --default-toolchain stable --profile minimal

WORKDIR /opt/hal

# Resolve deps first so this layer caches unless the lockfile changes: downloads
# managed CPython 3.14, every pinned wheel, and compiles the git forks.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Then the source + install the hal package itself.
COPY . .
RUN uv sync --locked

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
