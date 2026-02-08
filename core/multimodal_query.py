"""
マルチモーダルクエリエンジン
画像を文章中に埋め込んで回答を生成
"""
import base64
import io
import json
from typing import List, Dict, Any
from PIL import Image
import streamlit as st

from utils.logger import get_logger

logger = get_logger()


def image_to_base64(image: Image.Image) -> str:
    """PIL ImageをBase64エンコード"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def create_multimodal_prompt(query: str, nodes: List, image_cache) -> tuple:
    """
    マルチモーダルプロンプトを作成
    テキストと画像を組み合わせて、GPT-4 Visionに送信
    """
    text_parts = []
    image_documents = []
    
    text_parts.append(f"質問: {query}\n\n")
    text_parts.append("以下のコンテキストに基づいて質問に答えてください。\n")
    text_parts.append("画像がある場合は、画像の内容を参照して、文章中に「[画像1]」「[画像2]」のように番号で言及してください。\n\n")
    
    image_counter = 1
    
    for idx, node in enumerate(nodes):
        # テキスト部分
        text_parts.append(f"【ソース {idx + 1}】")
        text_parts.append(f"ファイル: {node.metadata.get('file_name', 'Unknown')}")
        text_parts.append(f"ページ: {node.metadata.get('page', '?')}")
        text_parts.append(f"\n{node.text}\n")
        
        # 画像部分
        if 'image_ids' in node.metadata:
            try:
                image_ids_str = node.metadata['image_ids']
                if isinstance(image_ids_str, str):
                    image_ids = json.loads(image_ids_str)
                else:
                    image_ids = image_ids_str
                
                for image_id in image_ids:
                    cached_data = image_cache.get_image(image_id)
                    if cached_data:
                        image = cached_data["image"]
                        metadata = cached_data["metadata"]
                        
                        # 画像情報を追加
                        text_parts.append(f"\n[画像{image_counter}]: {metadata.get('file_name')} - Page {metadata.get('page')}")
                        
                        # 画像ドキュメントを作成
                        image_documents.append({
                            "image": image,
                            "metadata": metadata,
                            "number": image_counter
                        })
                        
                        image_counter += 1
            except Exception as e:
                logger.warning(f"Failed to load images for node: {e}")
        
        text_parts.append("\n---\n")
    
    text_parts.append("\n回答の際は、関連する画像がある場合は「[画像1]」のように番号で言及してください。")
    text_parts.append("画像の内容を参照して、具体的に説明してください。")
    
    prompt_text = "\n".join(text_parts)
    
    return prompt_text, image_documents


def query_with_multimodal(
    index, 
    query_text: str, 
    similarity_top_k: int = 3,
    max_tokens: int = 2000,
    image_detail: str = "high",
    max_images: int = 5,
    response_mode: str = "compact",
    use_chat_history: bool = True,
    chat_history_length: int = 5,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int = None
) -> Dict[str, Any]:
    """
    マルチモーダルクエリを実行
    """
    logger.info(f"Executing multimodal query: {query_text[:50]}...")
    
    try:
        # 通常のRAG検索でノードを取得
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode
        )
        
        response = query_engine.query(query_text)
        source_nodes = response.source_nodes
        
        # 画像キャッシュ取得
        image_cache = st.session_state.image_cache
        
        # マルチモーダルプロンプト作成
        prompt_text, image_documents = create_multimodal_prompt(
            query_text, 
            source_nodes,
            image_cache
        )
        
        # GPT-4 Visionを使用（画像がある場合）
        if image_documents:
            logger.info(f"Using GPT-4 Vision with {len(image_documents)} images")
            
            # OpenAI APIを直接呼び出し
            import openai
            import os
            
            # 会話履歴を構築
            messages = []
            
            # 会話履歴を追加（有効な場合）
            if use_chat_history and "messages" in st.session_state:
                # 直近のNターンを取得（ユーザーとアシスタントのペア）
                recent_messages = st.session_state.messages[-(chat_history_length * 2):]
                for msg in recent_messages:
                    if msg["role"] == "user":
                        messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                    elif msg["role"] == "assistant":
                        messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
            
            # 現在のクエリを追加（画像付き）
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            })
            
            # 画像を追加（指定された最大数まで）
            for img_doc in image_documents[:max_images]:
                img_base64 = image_to_base64(img_doc["image"])
                messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": image_detail
                    }
                })
            
            # GPT-4 Vision API呼び出し（セッション状態のモデル設定を使用）
            llm_model = st.session_state.get("llm_model", "gpt-4o-mini")
            temperature = st.session_state.get("temperature", 0.1)
            
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            # APIパラメータを構築
            api_params = {
                "model": llm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            # Seedが指定されている場合のみ追加
            if seed is not None:
                api_params["seed"] = seed
            
            vision_response = client.chat.completions.create(**api_params)
            
            logger.info(f"Vision API call: model={llm_model}, temp={temperature}, top_p={top_p}, freq_pen={frequency_penalty}, pres_pen={presence_penalty}, seed={seed}, max_tokens={max_tokens}, images={len(image_documents[:max_images])}, detail={image_detail}, history={use_chat_history}({chat_history_length if use_chat_history else 0} turns)")
            
            answer_text = vision_response.choices[0].message.content
            
        else:
            # 画像がない場合は通常のLLM
            logger.info("No images found, using standard LLM")
            answer_text = response.response
        
        return {
            "answer": answer_text,
            "source_nodes": source_nodes,
            "image_documents": image_documents,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Multimodal query failed: {e}", exc_info=True)
        return {
            "answer": f"エラーが発生しました: {str(e)}",
            "source_nodes": [],
            "image_documents": [],
            "success": False,
            "error": str(e)
        }


def render_response_with_images(answer: str, image_documents: List[Dict]):
    """
    回答テキストを解析して、画像参照部分に実際の画像を埋め込む
    """
    import re
    
    # [画像1]、[画像2]などのパターンを検索
    pattern = r'\[画像(\d+)\]'
    
    # テキストを分割
    parts = re.split(pattern, answer)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # テキスト部分
            if part.strip():
                st.markdown(part)
        else:
            # 画像番号
            img_num = int(part)
            # 該当する画像を表示
            for img_doc in image_documents:
                if img_doc["number"] == img_num:
                    st.image(
                        img_doc["image"],
                        caption=f"画像{img_num}: {img_doc['metadata'].get('file_name')} - Page {img_doc['metadata'].get('page')}",
                        use_container_width=True
                    )
                    break
