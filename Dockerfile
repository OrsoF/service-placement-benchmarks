FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
ARG SERVICE_PLACEMENT_EXTRAS=benchmarks
RUN python -m pip install --no-cache-dir ".[${SERVICE_PLACEMENT_EXTRAS}]"
COPY data/ ./data/

ENTRYPOINT ["service-placement"]
CMD ["--help"]
