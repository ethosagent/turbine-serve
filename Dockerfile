# turbine-serve on NVIDIA DGX Spark. Base = NGC PyTorch: the only torch build for arm64 + sm_121.
# Dependencies come from uv.lock (exported to pip), minus torch and its CUDA wheels, which the base
# image already provides. Stage `deps` is the dev image (source mounted at /w); `release` is shipped.
#
#   dev:     docker build --target deps -t turbine-serve-dev .
#   release: docker build -t miteshdocker123/turbine-serve:0.1.0 .
FROM nvcr.io/nvidia/pytorch:26.07-py3 AS deps
COPY --from=ghcr.io/astral-sh/uv:0.7.11 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# torch and its CUDA wheels come from the base image (filtered out below); pyproject's CPU index
# only applies to uv on plain Linux hosts such as CI.
RUN uv export --frozen --group dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && grep -viE '^(torch|nvidia-|triton)' /tmp/requirements.txt > /tmp/requirements.filtered.txt \
 && uv pip install --system --break-system-packages --no-cache -r /tmp/requirements.filtered.txt

FROM deps AS release
COPY README.md ./
COPY turbine_serve ./turbine_serve
RUN uv pip install --system --break-system-packages --no-cache --no-deps .
EXPOSE 8000
ENTRYPOINT ["turbine", "serve", "--host", "0.0.0.0"]
