## generate pb
```bash
# use betterproto, don't use pydantic before it's compatible with v2
# move to multi stage docker build to avoid submit it to git
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=sentence_transformers/pb/ pb/sentence_transformers.embedding.proto
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=pb/ pb/sentence_transformers.embedding.proto
```

## local test
```bash
cd grpc_servers/sentence_transformers
pytest -vv tests/sentence_transformers_test.py
```

## configuration
|environment|default|comment|
|----|----|----|
|SENTENCE_TRANSFORMERS_MODEL|hf-internal-testing/tiny-random-bert|huggingface compatible model name|
|SENTENCE_TRANSFORMERS_HOST|0.0.0.0|listen host|
|SENTENCE_TRANSFORMERS_PORT|50051|listen port|
