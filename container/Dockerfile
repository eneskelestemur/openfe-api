# openfe-api: CLI and HTTP service in one image.
#
# Build (tag anything you keep with the version from pyproject.toml):
#   docker build -f container/Dockerfile -t openfe-api:latest .
#
# Run the service:
#   docker run --rm -p 8000:8000 -v /data:/data -e OPENFE_API_ROOT=/data/campaigns \
#       openfe-api:latest
#
# Run the CLI instead:
#   docker run --rm -v /data:/data openfe-api:latest openfe-api status /data/campaigns/my_run
#
# Simulations need a GPU: pass --gpus all. Submitting to a host scheduler from inside the
# container needs its client, configuration and authentication socket bound in.
FROM mambaorg/micromamba:2.0.5

LABEL org.opencontainers.image.title="openfe-api" \
      org.opencontainers.image.description="Run OpenFE free energy experiments on HPC and server VMs." \
      org.opencontainers.image.source="https://github.com/eneskelestemur/openfe-api" \
      org.opencontainers.image.licenses="MIT"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER container/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt/openfe-api
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
RUN pip install --no-deps --no-cache-dir .

ENV OPENFE_API_ROOT=/data/campaigns \
    OPENFE_API_PROCESSORS=1 \
    MPLCONFIGDIR=/tmp/openfe-api-matplotlib \
    XDG_CACHE_HOME=/tmp/openfe-api-cache

EXPOSE 8000
WORKDIR /data

ENTRYPOINT ["tini", "--", "/usr/local/bin/_entrypoint.sh"]
CMD ["uvicorn", "openfe_api.service:app", "--host", "0.0.0.0", "--port", "8000"]
