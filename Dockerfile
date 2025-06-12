FROM python:3.11-slim as base
WORKDIR /tmp
RUN pip install poetry && poetry self add poetry-plugin-export
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# New stage for heavy dependencies
FROM python:3.11-slim as heavy-deps
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=base /tmp/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

FROM heavy-deps as dev
ARG GIT_HASH
ARG GIT_BRANCH
ARG GIT_TAG
ENV GIT_HASH=${GIT_HASH}
ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_TAG=${GIT_TAG}
ENV PYTHONPATH=./
WORKDIR /app
COPY . .
ENTRYPOINT ["python", "app/run.py"]