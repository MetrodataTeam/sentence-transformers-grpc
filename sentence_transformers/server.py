"""
https://www.philschmid.de/optimize-sentence-transformers
https://huggingface.co/docs/optimum/en/onnxruntime/usage_guides/pipelines
"""
import asyncio
import json
import logging
from base64 import b64encode
from time import time
from typing import List

import onnxruntime
import torch
from grpclib.health.service import Health
from grpclib.reflection.service import ServerReflection
from grpclib.server import Server
from grpclib.utils import graceful_exit
from huggingface_hub import try_to_load_from_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from sentence_transformers.models import Pooling
from transformers import AutoTokenizer
from transformers import Pipeline

from pb.sentence_transformers.embedding import CountTokensMessage
from pb.sentence_transformers.embedding import CountTokensResponse
from pb.sentence_transformers.embedding import CountTokensResponsePart
from pb.sentence_transformers.embedding import EmbeddingMessage
from pb.sentence_transformers.embedding import EmbeddingMessageResponse
from pb.sentence_transformers.embedding import EmbeddingServiceBase
from pb.sentence_transformers.embedding import EncodingFormat
from pb.sentence_transformers.embedding import SplitTextMessage
from pb.sentence_transformers.embedding import SplitTextResponse
from pb.sentence_transformers.embedding import SplitTextResponsePart


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_prefix='sentence_transformers_')
  model: str = Field('hf-internal-testing/tiny-random-bert',
                     description='huggingface hub compatible model id')
  host: str = Field('0.0.0.0', description='listen host')
  port: int = Field(50051, description='listen port')
  # TODO(Deo): use it if we implement internal load balancing
  # concurrent: PositiveInt = Field(1, description='size of the tread pool')


class SentenceEmbeddingPipeline(Pipeline):

  def _sanitize_parameters(self, pool=None, **kwargs):
    # TODO(Deo): distinguish between initial and runtime pool
    if pool is not None:
      self.pool = pool
    return {}, {}, {'pool': self.pool}

  def preprocess(self, inputs):
    encoded_inputs = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')
    return encoded_inputs

  def _forward(self, model_inputs):
    outputs = self.model(**model_inputs)
    return {
        "token_embeddings": outputs[0],
        "attention_mask": model_inputs["attention_mask"]
    }

  def postprocess(self, model_outputs, pool: Pooling):
    # Perform pooling
    sentence_embeddings = pool.forward(model_outputs)
    return sentence_embeddings['sentence_embedding']


class EmbeddingService(EmbeddingServiceBase):

  def __init__(self, model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Use device: %s", device)

    onnxproviders = onnxruntime.get_available_providers()

    if device == "cpu":
      fast_onnxprovider = "CPUExecutionProvider"
    else:
      if "CUDAExecutionProvider" not in onnxproviders:
        logging.warning("Using CPU. Try installing 'onnxruntime-gpu'.")
        fast_onnxprovider = "CPUExecutionProvider"
      else:
        fast_onnxprovider = "CUDAExecutionProvider"
    # TODO(Deo): save the onnx model and load from saved one next time
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(
        model,
        provider=fast_onnxprovider,
        export=True,
    )
    path = try_to_load_from_cache(model, 'config.json')
    with open(path) as f:
      config = json.load(f)
    # max input tokens
    self.max_seq_length = config['max_position_embeddings']
    # embedding dimesion
    self.hidden_size = config['hidden_size']
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    # TODO(Deo): read from config, default to mean pooling
    pool = Pooling(self.hidden_size, pooling_mode='mean')
    self.pipeline = SentenceEmbeddingPipeline(model=onnx_model,
                                              tokenizer=self.tokenizer,
                                              pool=pool)
    # TODO(Deo): make this configurable and use regex
    # currently using this will attpend the regex to text
    # self.separators = [r'\n\n|\n|！|？|｡|。| ', r'']
    self.separators = ['\n\n', '\n', '！', '？', '｡', '。', ' ', '']

  async def embedding(self,
                      request: EmbeddingMessage) -> EmbeddingMessageResponse:
    start = time()
    if not request.truncate:
      errors = []
      # TODO(Deo): find a way to strip tensor padded zeros
      for index, i in enumerate(request.messages):
        tokens = self.tokenizer([i])['input_ids'][0]
        # TODO(Deo): make tokenizer not to truncate input
        if len(tokens) == self.max_seq_length:
          errors.append(index)
      if errors:
        return EmbeddingMessageResponse(row=0, col=0, errors=errors)
    embeddings = self.pipeline(request.messages)
    if request.encoding_format == EncodingFormat.BASE64:
      res = EmbeddingMessageResponse(
          row=len(request.messages),
          col=self.hidden_size,
          b64_data=[
              b64encode(i.numpy().tobytes()).decode() for i in embeddings
          ])
    else:
      res = EmbeddingMessageResponse(
          row=len(request.messages),
          col=self.hidden_size,
          data=torch.cat([i.flatten() for i in embeddings]).tolist())
    logging.info(
        'finish %d sentences in %.3f ms',
        len(request.messages),
        (time() - start) * 1000,
    )
    return res

  async def count_tokens(self,
                         request: CountTokensMessage) -> CountTokensResponse:
    start = time()
    res = CountTokensResponse()
    tokens = self.tokenizer(request.messages)['input_ids']
    for i in tokens:
      res.parts.append(
          CountTokensResponsePart(
              tokens=len(i),
              needs_truncation=len(i) > self.max_seq_length,
          ))
    logging.info(
        'finish %d count tokens in %.3f ms',
        len(request.messages),
        (time() - start) * 1000,
    )
    return res

  async def split_text(self, request: SplitTextMessage) -> SplitTextResponse:
    start = time()
    if request.chunk_size:
      chunk_size = min(request.chunk_size, self.max_seq_length)
    else:
      chunk_size = self.max_seq_length
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=self.tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=request.chunk_overlap,
        separators=self.separators,
        is_separator_regex=False,
        keep_separator=False,
    )
    res = SplitTextResponse()
    for i in request.messages:
      texts = text_splitter.split_text(i)
      res.parts.append(SplitTextResponsePart(parts=texts))
    logging.info(
        'finish %d split text in %.3f ms',
        len(request.messages),
        (time() - start) * 1000,
    )
    return res


def get_services(model: str) -> List:
  return ServerReflection.extend([EmbeddingService(model), Health()])


async def serve(settings: Settings):
  server = Server(get_services(settings.model))
  with graceful_exit([server]):
    await server.start(settings.host, settings.port)
    logging.info(
        'listen on %s:%d, using %s',
        settings.host,
        settings.port,
        settings.model,
    )
    await server.wait_closed()
    logging.info('Goodbye!')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  _settings = Settings()
  asyncio.run(serve(_settings))
