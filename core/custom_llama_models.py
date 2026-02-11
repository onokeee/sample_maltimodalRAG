"""
LlamaIndex 用カスタム Embedding / LLM クラス

openai ライブラリを使わず、openai_client.py（requests ベース）経由で
API を呼び出す。これにより公式 OpenAI / Azure / 社内 API を
同じコードで利用できる。
"""
from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from core.openai_client import call_embedding_api, call_chat_api
from utils.logger import get_logger

logger = get_logger()


# --------------------------------------------------------------------------- #
#  カスタム Embedding
# --------------------------------------------------------------------------- #
class CustomOpenAIEmbedding(BaseEmbedding):
    """
    requests ベースの OpenAI Embedding。
    openai_client.call_embedding_api() を使用。
    """

    model_name: str = Field(default="text-embedding-3-small", description="Embedding model name")

    def __init__(self, model: str = "text-embedding-3-small", **kwargs: Any):
        super().__init__(model_name=model, **kwargs)

    class Config:
        arbitrary_types_allowed = True

    def _get_query_embedding(self, query: str) -> List[float]:
        """クエリ用 Embedding（単一テキスト）"""
        embeddings = call_embedding_api(texts=[query], model=self.model_name)
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """ドキュメント用 Embedding（単一テキスト）"""
        embeddings = call_embedding_api(texts=[text], model=self.model_name)
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """ドキュメント用 Embedding（バッチ）"""
        # API の制限を考慮して分割（最大 2048 テキスト / リクエスト）
        batch_size = 2048
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = call_embedding_api(texts=batch, model=self.model_name)
            all_embeddings.extend(embeddings)
        return all_embeddings

    # 非同期版（同期版にフォールバック）
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


# --------------------------------------------------------------------------- #
#  カスタム LLM
# --------------------------------------------------------------------------- #
class CustomOpenAILLM(CustomLLM):
    """
    requests ベースの OpenAI Chat LLM。
    openai_client.call_chat_api() を使用。
    """

    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")
    temperature: float = Field(default=0.1, description="Temperature")
    max_tokens: int = Field(default=2000, description="Max tokens")

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    class Config:
        arbitrary_types_allowed = True

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """テキスト生成（LlamaIndex が RAG で使用）"""
        messages = [{"role": "user", "content": prompt}]

        answer = call_chat_api(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return CompletionResponse(text=answer)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """ストリーミング生成（非対応 → 一括で返す）"""
        response = self.complete(prompt, **kwargs)
        # ストリーミング風に1回で返す
        def gen():
            yield response
        return gen()
