# Production image for the HAL netplay API and GPU runner.
FROM vastai/base-image:cuda-13.0.2-auto AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    PATH=/root/.cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config \
      libasound2-dev libegl-dev libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
      --default-toolchain stable --profile minimal

WORKDIR /opt/hal
COPY pyproject.toml uv.lock README.md ./
COPY hal ./hal
RUN uv sync --locked --extra netplay-server --no-editable

FROM vastai/base-image:cuda-13.0.2-auto

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl xvfb libfuse2t64 libvulkan1 \
      libasound2t64 libegl1 libgl1 libusb-1.0-0 \
      libgtk-3-0t64 libglib2.0-0t64 libgdk-pixbuf-2.0-0 \
      libpangocairo-1.0-0 libx11-6 libxcursor1 libxi6 libxinerama1 \
      libxkbcommon0 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /.uv/python_install /.uv/python_install
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hal /opt/hal
COPY docker/entrypoint.sh /usr/local/bin/hal-netplay-entrypoint
RUN chmod +x /usr/local/bin/hal-netplay-entrypoint

WORKDIR /opt/hal
ENTRYPOINT ["/usr/local/bin/hal-netplay-entrypoint"]
CMD ["hal-netplay-api", "--host", "0.0.0.0"]
