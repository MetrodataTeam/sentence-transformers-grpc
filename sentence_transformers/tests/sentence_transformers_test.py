from base64 import b64decode

import numpy as np
import pytest
from grpclib.health.v1.health_grpc import HealthStub
from grpclib.health.v1.health_pb2 import HealthCheckRequest
from grpclib.health.v1.health_pb2 import HealthCheckResponse
from grpclib.reflection.v1.reflection_grpc import ServerReflectionStub
from grpclib.reflection.v1.reflection_pb2 import ServerReflectionRequest
from grpclib.testing import ChannelFor
from mdt_util.test import assert_equal
from server import Settings
from server import get_services

from pb.sentence_transformers.embedding import CountTokensMessage
from pb.sentence_transformers.embedding import CountTokensResponsePart
from pb.sentence_transformers.embedding import EmbeddingMessage
from pb.sentence_transformers.embedding import EmbeddingServiceStub
from pb.sentence_transformers.embedding import EncodingFormat
from pb.sentence_transformers.embedding import SplitTextMessage
from pb.sentence_transformers.embedding import SplitTextResponsePart


@pytest.fixture(scope='module', autouse=True)
def anyio_backend():
  return 'asyncio'


async def test_embedding_service():
  settings = Settings()
  services = get_services(settings.model)
  expect = [
      0.8778659701347351, -0.5448046922683716, 0.3896498680114746,
      0.19267699122428894, -0.6208742260932922, 0.6843146085739136,
      -0.03016902133822441, -0.40138888359069824, -0.8664363622665405,
      0.4803892970085144, 0.31438177824020386, 0.4647614061832428,
      0.003958487417548895, 0.4428254961967468, -0.6020190119743347,
      0.014086651615798473, -0.15043002367019653, -0.7715902328491211,
      -0.3880080580711365, 1.2349672317504883, 0.8893807530403137,
      0.11006786674261093, -0.5883675813674927, 0.6363809108734131,
      0.26165053248405457, 0.9332454800605774, -0.14919307827949524,
      -0.9826553463935852, 0.1642739176750183, -0.1309153139591217,
      -0.8565775752067566, -1.0114481449127197, 0.6645352840423584,
      -0.9539824724197388, -0.053713928908109665, 0.13536418974399567,
      -0.5071420073509216, 0.2011839896440506, -0.32303714752197266,
      0.23485179245471954, -0.5548384785652161, 0.9774921536445618,
      0.9987732172012329, 0.12581613659858704, 0.323246568441391,
      0.8223288655281067, -0.6005865335464478, 0.08547230064868927,
      -0.48118752241134644, -0.8802777528762817, -0.5932547450065613,
      1.11785888671875, 1.0007895231246948, 0.6238759756088257,
      -0.8345882296562195, 0.247264102101326, 0.02627432346343994,
      0.6484103202819824, 0.3002376854419708, -0.5706024765968323,
      0.14048168063163757, -0.8811043500900269, -0.8896624445915222,
      -0.5502783060073853
  ]
  async with ChannelFor(services) as channel:
    stub = EmbeddingServiceStub(channel)
    health = HealthStub(channel)
    reflection = ServerReflectionStub(channel)

    # float
    response = await stub.embedding(
        EmbeddingMessage(encoding_format=EncodingFormat.FLOAT,
                         messages=['aaa', 'bbb']))
    assert response.row == 2
    assert response.col == 32
    assert_equal(response.data, expect)

    # base64
    response = await stub.embedding(
        EmbeddingMessage(encoding_format=EncodingFormat.BASE64,
                         messages=['aaa', 'bbb']))
    assert response.row == 2
    assert response.col == 32
    actual = []
    for i in response.b64_data:
      actual.extend(np.frombuffer(b64decode(i), dtype='float32').tolist())
    assert_equal(actual, expect)

    # count tokens
    response = await stub.count_tokens(
        CountTokensMessage(messages=[
            'here comes the great team to solve the hardest problem.',
            '我们明天要去秋游啦! 天气还不错'
        ]))
    assert response.parts == [
        CountTokensResponsePart(tokens=48, needs_truncation=False),
        CountTokensResponsePart(tokens=17, needs_truncation=False),
    ]

    # split text
    # auto, 0
    # fixed, 20
    for chunk_size, (first, second) in zip(
        [0, 20],
        [(
            ['here comes the great team to solve the hardest problem.'],
            ['我们明天要去秋游啦! 天气还不错'],
        ),
         (
             [
                 'here comes',
                 'the great',
                 'team to',
                 'solve the',
                 'hardest',
                 'problem.',
             ],
             ['我们明天要去秋游啦!', '天气还不错'],
         )],
    ):
      response = await stub.split_text(
          SplitTextMessage(
              messages=[
                  'here comes the great team to solve the hardest problem.',
                  '我们明天要去秋游啦! 天气还不错'
              ],
              chunk_size=chunk_size,
              chunk_overlap=1,
          ))
      assert response.parts == [
          SplitTextResponsePart(parts=first),
          SplitTextResponsePart(parts=second),
      ]

    # health
    response = await health.Check(HealthCheckRequest())
    assert response.status == HealthCheckResponse.SERVING

    # reflection
    response = await reflection.ServerReflectionInfo(
        [ServerReflectionRequest(file_containing_symbol='Embedding')])
    assert len(response) == 1
    # TODO(Deo): it's not found at the moment
    #   https://github.com/danielgtaylor/python-betterproto/issues/443
    # assert response[0].name == ''
    # assert response[0].package == ''
