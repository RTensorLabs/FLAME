# Hugging Face Docker Space for the FLAME 30M Rust model.
# Stage 1 builds the `flame` CLI (CPU only); stage 2 runs a small Gradio app around it.
# Both stages use the same base image, so the binary's glibc always matches.

FROM python:3.12-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.1
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY flame-src/ .
RUN cargo build --release --locked -p flame-cli

FROM python:3.12-slim
# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=build /src/target/release/flame /usr/local/bin/flame
COPY --chown=user model/ /app/model/
COPY --chown=user app.py .
USER user
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    FLAME_MODEL_DIR=/app/model
EXPOSE 7860
CMD ["python", "app.py"]
