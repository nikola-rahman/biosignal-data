# Runs `biosignal-data` on any machine, e.g. a server close to your storage, so that
# downloads do not pass through your own computer.
#
#   docker build -t biosignal-data .
#   docker run --rm -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
#       biosignal-data download --all --dest s3://bucket --endpoint https://...
#
# The image has every tool the fetchers call: rsync and ssh (TUH), the NSRR Ruby gem,
# and the Kaggle CLI (a Python dependency).

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends rsync openssh-client ruby ca-certificates \
    && gem install --no-document nsrr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY catalog ./catalog
RUN uv sync --locked --no-dev

ENV PATH="/opt/venv/bin:$PATH" \
    BIOSIGNAL_CATALOG=/app/catalog \
    BIOSIGNAL_DATA_CACHE=/cache
# Staged sources (rsync, nsrr, kaggle) need local disk: mount a volume at /staging and
# pass --staging-dir /staging.
VOLUME ["/staging", "/cache"]
ENTRYPOINT ["biosignal-data"]
CMD ["--help"]
