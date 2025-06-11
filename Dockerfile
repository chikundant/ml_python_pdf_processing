FROM python:3.11-slim as base

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM base as base-requirements

WORKDIR /app

COPY --from=base /tmp/requirements.txt /requirements.txt

ENV PYTHONPATH=./

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

FROM base-requirements as dev
ARG GIT_HASH
ARG GIT_BRANCH
ARG GIT_TAG
ENV GIT_HASH=${GIT_HASH}
ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_TAG=${GIT_TAG}

WORKDIR /app

COPY . .

ENTRYPOINT ["python", "app/run.py"]
