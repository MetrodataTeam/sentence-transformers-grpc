# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: sentence_transformers.embedding.proto
# plugin: python-betterproto
# This file has been @generated

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
)

import betterproto
import grpclib
from betterproto.grpc.grpclib_server import ServiceBase


if TYPE_CHECKING:
    import grpclib.server
    from betterproto.grpc.grpclib_client import MetadataLike
    from grpclib.metadata import Deadline


class EncodingFormat(betterproto.Enum):
    FLOAT = 0
    BASE64 = 1


@dataclass(eq=False, repr=False)
class EmbeddingMessage(betterproto.Message):
    encoding_format: "EncodingFormat" = betterproto.enum_field(1)
    messages: List[str] = betterproto.string_field(2)
    truncate: Optional[bool] = betterproto.bool_field(
        3, optional=True, group="_truncate"
    )


@dataclass(eq=False, repr=False)
class EmbeddingMessageResponse(betterproto.Message):
    row: int = betterproto.uint32_field(1)
    col: int = betterproto.uint32_field(2)
    data: List[float] = betterproto.float_field(3)
    b64_data: List[str] = betterproto.string_field(4)
    errors: List[int] = betterproto.uint32_field(5)


@dataclass(eq=False, repr=False)
class CountTokensMessage(betterproto.Message):
    messages: List[str] = betterproto.string_field(1)


@dataclass(eq=False, repr=False)
class CountTokensResponsePart(betterproto.Message):
    tokens: int = betterproto.uint32_field(1)
    needs_truncation: bool = betterproto.bool_field(2)


@dataclass(eq=False, repr=False)
class CountTokensResponse(betterproto.Message):
    parts: List["CountTokensResponsePart"] = betterproto.message_field(1)


@dataclass(eq=False, repr=False)
class SplitTextMessage(betterproto.Message):
    messages: List[str] = betterproto.string_field(1)
    chunk_size: int = betterproto.uint32_field(2)
    chunk_overlap: int = betterproto.uint32_field(3)


@dataclass(eq=False, repr=False)
class SplitTextResponsePart(betterproto.Message):
    parts: List[str] = betterproto.string_field(1)


@dataclass(eq=False, repr=False)
class SplitTextResponse(betterproto.Message):
    parts: List["SplitTextResponsePart"] = betterproto.message_field(1)


class EmbeddingServiceStub(betterproto.ServiceStub):
    async def embedding(
        self,
        embedding_message: "EmbeddingMessage",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "EmbeddingMessageResponse":
        return await self._unary_unary(
            "/sentence_transformers.embedding.EmbeddingService/embedding",
            embedding_message,
            EmbeddingMessageResponse,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def count_tokens(
        self,
        count_tokens_message: "CountTokensMessage",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "CountTokensResponse":
        return await self._unary_unary(
            "/sentence_transformers.embedding.EmbeddingService/count_tokens",
            count_tokens_message,
            CountTokensResponse,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def split_text(
        self,
        split_text_message: "SplitTextMessage",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "SplitTextResponse":
        return await self._unary_unary(
            "/sentence_transformers.embedding.EmbeddingService/split_text",
            split_text_message,
            SplitTextResponse,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )


class EmbeddingServiceBase(ServiceBase):
    async def embedding(
        self, embedding_message: "EmbeddingMessage"
    ) -> "EmbeddingMessageResponse":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def count_tokens(
        self, count_tokens_message: "CountTokensMessage"
    ) -> "CountTokensResponse":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def split_text(
        self, split_text_message: "SplitTextMessage"
    ) -> "SplitTextResponse":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def __rpc_embedding(
        self,
        stream: "grpclib.server.Stream[EmbeddingMessage, EmbeddingMessageResponse]",
    ) -> None:
        request = await stream.recv_message()
        response = await self.embedding(request)
        await stream.send_message(response)

    async def __rpc_count_tokens(
        self, stream: "grpclib.server.Stream[CountTokensMessage, CountTokensResponse]"
    ) -> None:
        request = await stream.recv_message()
        response = await self.count_tokens(request)
        await stream.send_message(response)

    async def __rpc_split_text(
        self, stream: "grpclib.server.Stream[SplitTextMessage, SplitTextResponse]"
    ) -> None:
        request = await stream.recv_message()
        response = await self.split_text(request)
        await stream.send_message(response)

    def __mapping__(self) -> Dict[str, grpclib.const.Handler]:
        return {
            "/sentence_transformers.embedding.EmbeddingService/embedding": grpclib.const.Handler(
                self.__rpc_embedding,
                grpclib.const.Cardinality.UNARY_UNARY,
                EmbeddingMessage,
                EmbeddingMessageResponse,
            ),
            "/sentence_transformers.embedding.EmbeddingService/count_tokens": grpclib.const.Handler(
                self.__rpc_count_tokens,
                grpclib.const.Cardinality.UNARY_UNARY,
                CountTokensMessage,
                CountTokensResponse,
            ),
            "/sentence_transformers.embedding.EmbeddingService/split_text": grpclib.const.Handler(
                self.__rpc_split_text,
                grpclib.const.Cardinality.UNARY_UNARY,
                SplitTextMessage,
                SplitTextResponse,
            ),
        }
