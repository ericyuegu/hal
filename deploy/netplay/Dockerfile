# Production image for the HAL netplay GPU runner.
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
COPY vendor ./vendor
RUN uv sync --locked --no-editable

FROM vastai/base-image:cuda-13.0.2-auto

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl fonts-dejavu-core procps pulseaudio x11-xserver-utils xvfb \
      obs-studio=30.2.3+dfsg-3~bpo24.04.1 libobs0=30.2.3+dfsg-3~bpo24.04.1 \
      openbox mesa-utils xauth x11-utils tesseract-ocr \
      libfuse2t64 libvulkan1 vulkan-tools \
      libasound2t64 libegl1 libgl1 libopengl0 libusb-1.0-0 \
      libgtk-3-0t64 libglib2.0-0t64 libgdk-pixbuf-2.0-0 \
      libpangocairo-1.0-0 libx11-6 libxcursor1 libxi6 libxinerama1 \
      libxkbcommon0 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA Container Toolkit injects the driver libraries; EGL also needs its
# vendor registration to keep OBS from selecting Mesa's software renderer.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=builder /.uv/python_install /.uv/python_install
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hal /opt/hal
COPY scripts/play_ranked.py /opt/hal/scripts/play_ranked.py
COPY docker/entrypoint.sh /usr/local/bin/hal-netplay-entrypoint
RUN chmod +x /usr/local/bin/hal-netplay-entrypoint

ARG HAL_GIT_SHA
RUN test -n "$HAL_GIT_SHA"
ENV HAL_GIT_SHA="$HAL_GIT_SHA"
LABEL org.opencontainers.image.revision="$HAL_GIT_SHA"

WORKDIR /opt/hal
ENTRYPOINT ["/usr/local/bin/hal-netplay-entrypoint"]
CMD ["hal-netplay-runner", "--slots", "1", "--compiled"]
