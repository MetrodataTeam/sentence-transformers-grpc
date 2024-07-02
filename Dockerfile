FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ARG target=/mdt/run

# 镜像commit信息
ARG COMMIT_TAG
ARG COMMIT_HASH
ENV COMMIT_TAG=$COMMIT_TAG
ENV COMMIT_HASH=$COMMIT_HASH

ARG TARGETPLATFORM=linux/amd64
ARG TARGETARCH=amd64

WORKDIR ${target}

RUN pip install --no-cache-dir -i ${PYPI} poetry==1.8.3 && \
    poetry config virtualenvs.create false

ADD sentence_transformers/pyproject.toml sentence_transformers/poetry.lock ${target}/

RUN poetry install --no-cache

COPY sentence_transformers/pb/ ${target}/pb/
COPY sentence_transformers/*.py ${target}/

RUN python -m compileall ${target}
CMD ["python", "server.py"]
EXPOSE 18908
