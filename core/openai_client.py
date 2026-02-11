"""
OpenAI API クライアントモジュール（requests ベース）
ChatGPT / GPT-4 Vision / Embeddings API との連携を一元管理

openai ライブラリに依存せず、HTTP リクエストのみで通信するため、
公式 OpenAI API / Azure OpenAI / 社内 ChatGPT API など
任意のエンドポイントに切り替え可能。

■ 設定方法（.env または環境変数）
  OPENAI_API_KEY       ... APIキー（必須）
  OPENAI_API_BASE_URL  ... エンドポイントURL（省略時: https://api.openai.com/v1）

  社内APIの例:
    OPENAI_API_BASE_URL=https://your-company-api.example.com/v1

  Azure OpenAIの例:
    OPENAI_API_BASE_URL=https://{リソース名}.openai.azure.com/openai/deployments/{デプロイ名}
    ※ Azure の場合は認証ヘッダーも異なるため OPENAI_API_AUTH_TYPE=azure を設定

■ Azure OpenAI 対応
  OPENAI_API_AUTH_TYPE  ... 認証方式（省略時: bearer）
    bearer  → Authorization: Bearer {api_key}  （公式OpenAI / 社内API）
    azure   → api-key: {api_key}               （Azure OpenAI）
  OPENAI_API_VERSION    ... Azure の api-version（例: 2024-02-01）
"""
import os
import requests
from typing import List, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger()

# --------------------------------------------------------------------------- #
#  定数 / デフォルト値
# --------------------------------------------------------------------------- #
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_CHAT_COMPLETIONS_PATH = "/chat/completions"
_EMBEDDINGS_PATH = "/embeddings"
_REQUEST_TIMEOUT = 120  # 秒


# --------------------------------------------------------------------------- #
#  内部ヘルパー
# --------------------------------------------------------------------------- #
def _get_api_key() -> str:
    """環境変数から API キーを取得"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY が設定されていません")
    return api_key


def _get_base_url() -> str:
    """
    環境変数からエンドポイントの Base URL を取得。
    未設定の場合は公式 OpenAI API を使用。
    """
    return os.environ.get("OPENAI_API_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _get_auth_type() -> str:
    """認証方式を取得（bearer / azure）"""
    return os.environ.get("OPENAI_API_AUTH_TYPE", "bearer").lower()


def _get_headers() -> Dict[str, str]:
    """認証方式に応じたHTTPヘッダーを構築"""
    api_key = _get_api_key()
    auth_type = _get_auth_type()

    headers = {"Content-Type": "application/json"}

    if auth_type == "azure":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def _build_url(path: str) -> str:
    """エンドポイントURLを構築（Azure の api-version 対応）"""
    base_url = _get_base_url()
    url = f"{base_url}{path}"

    # Azure の場合は api-version クエリパラメータを付与
    if _get_auth_type() == "azure":
        api_version = os.environ.get("OPENAI_API_VERSION", "2024-02-01")
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}api-version={api_version}"

    return url


def _handle_error(resp: requests.Response) -> None:
    """APIエラーレスポンスを処理"""
    error_detail = ""
    try:
        error_body = resp.json()
        error_detail = error_body.get("error", {}).get("message", resp.text)
    except Exception:
        error_detail = resp.text
    logger.error(f"API error ({resp.status_code}): {error_detail}")
    raise requests.HTTPError(
        f"APIエラー ({resp.status_code}): {error_detail}",
        response=resp,
    )


# --------------------------------------------------------------------------- #
#  公開 API - Chat Completions
# --------------------------------------------------------------------------- #
def build_chat_messages(
    prompt_text: str,
    image_base64_list: List[str],
    image_detail: str = "high",
    max_images: int = 5,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    ChatGPT API に送信する messages 配列を構築

    Args:
        prompt_text: プロンプトテキスト
        image_base64_list: Base64エンコードされた画像のリスト
        image_detail: 画像詳細レベル ("high", "low", "auto")
        max_images: 最大画像数
        chat_history: 過去の会話履歴 [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        OpenAI 互換の messages パラメータ
    """
    messages: List[Dict[str, Any]] = []

    # 会話履歴を追加
    if chat_history:
        for msg in chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    # 現在のクエリ（テキスト + 画像）
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text}
    ]

    for img_b64 in image_base64_list[:max_images]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": image_detail,
            },
        })

    messages.append({"role": "user", "content": content})

    return messages


def call_chat_api(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = None,
) -> str:
    """
    Chat Completions API を HTTP POST で呼び出し、回答テキストを返す。

    Args:
        messages: メッセージ配列
        model: 使用モデル
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        top_p: Top P
        frequency_penalty: 頻度ペナルティ
        presence_penalty: 存在ペナルティ
        seed: シード値（再現性）

    Returns:
        アシスタントの回答テキスト
    """
    url = _build_url(_CHAT_COMPLETIONS_PATH)
    headers = _get_headers()

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }

    if seed is not None:
        payload["seed"] = seed

    logger.info(
        f"Chat API call: url={url}, model={model}, temp={temperature}, "
        f"top_p={top_p}, freq_pen={frequency_penalty}, "
        f"pres_pen={presence_penalty}, seed={seed}, max_tokens={max_tokens}"
    )

    resp = requests.post(url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT)

    if resp.status_code != 200:
        _handle_error(resp)

    data = resp.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected response structure: {data}")
        raise ValueError(f"APIレスポンスの解析に失敗しました: {e}")

    logger.info(f"Chat API response received ({len(answer)} chars)")
    return answer


# --------------------------------------------------------------------------- #
#  公開 API - Embeddings
# --------------------------------------------------------------------------- #
def call_embedding_api(
    texts: List[str],
    model: str = "text-embedding-3-small",
) -> List[List[float]]:
    """
    Embeddings API を HTTP POST で呼び出し、ベクトルのリストを返す。

    Args:
        texts: 埋め込みたいテキストのリスト
        model: 使用するEmbeddingモデル

    Returns:
        各テキストに対応するベクトル（float のリスト）のリスト
    """
    url = _build_url(_EMBEDDINGS_PATH)
    headers = _get_headers()

    payload: Dict[str, Any] = {
        "model": model,
        "input": texts,
    }

    logger.info(f"Embedding API call: url={url}, model={model}, texts={len(texts)}")

    resp = requests.post(url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT)

    if resp.status_code != 200:
        _handle_error(resp)

    data = resp.json()

    try:
        # レスポンスの data 配列を index 順にソートしてベクトルを取り出す
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in embeddings_data]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Unexpected embedding response structure: {data}")
        raise ValueError(f"Embedding APIレスポンスの解析に失敗しました: {e}")

    logger.info(f"Embedding API response: {len(embeddings)} vectors, dim={len(embeddings[0]) if embeddings else 0}")
    return embeddings
