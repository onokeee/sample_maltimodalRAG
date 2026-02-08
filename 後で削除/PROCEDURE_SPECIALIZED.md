# 作業手順書特化型RAGシステム 設計書

## 🎯 目的
作業手順書に特化することで、以下を実現：
1. **手順の順序理解**: ステップ1→2→3の流れを正確に把握
2. **図表との紐付け**: 「図1参照」などを正確に理解
3. **注意事項の強調**: 警告・注意マークを見逃さない
4. **チェックリスト生成**: 手順から自動でチェックリスト作成

## 📋 改善ポイント

### 1. ドキュメント構造の認識強化

#### 現状の問題
```python
# 単純なページ分割
doc = Document(text=page_text, metadata={"page": 1})
```

#### 改善案
```python
# 手順書構造を解析
doc = Document(
    text=page_text,
    metadata={
        "page": 1,
        "section_type": "手順",  # 概要/手順/注意事項/参考
        "step_numbers": [1, 2, 3],  # このページに含まれるステップ番号
        "has_warnings": True,  # 警告の有無
        "referenced_figures": ["図1", "図2"],  # 参照される図
        "checklist_items": ["確認", "テスト"]  # チェック項目
    }
)
```

### 2. 手順番号の認識

#### パターン検出
```python
import re

def extract_step_numbers(text):
    """手順番号を抽出"""
    patterns = [
        r'ステップ[\s]*(\d+)',
        r'手順[\s]*(\d+)',
        r'^(\d+)[\.)．]',  # 1. 2. 3.
        r'【(\d+)】',
    ]
    
    step_numbers = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        step_numbers.extend([int(m.group(1)) for m in matches])
    
    return sorted(set(step_numbers))
```

### 3. 注意事項の検出

#### キーワードベースの重要度付け
```python
def detect_warnings(text):
    """警告・注意事項を検出"""
    warning_keywords = {
        "critical": ["危険", "禁止", "絶対に", "必ず"],
        "warning": ["注意", "警告", "重要"],
        "caution": ["確認", "注記", "留意"]
    }
    
    warnings = []
    for level, keywords in warning_keywords.items():
        for keyword in keywords:
            if keyword in text:
                warnings.append({
                    "level": level,
                    "keyword": keyword,
                    "context": extract_context(text, keyword)
                })
    
    return warnings
```

### 4. 図表参照の解析

#### 参照関係の抽出
```python
def extract_figure_references(text):
    """図表参照を抽出"""
    patterns = [
        r'図[\s]*(\d+)',
        r'表[\s]*(\d+)',
        r'Fig\.?[\s]*(\d+)',
        r'画像[\s]*(\d+)',
    ]
    
    references = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            references.append({
                "type": "figure" if "図" in match.group(0) else "table",
                "number": int(match.group(1)),
                "context": extract_context(text, match.group(0))
            })
    
    return references
```

### 5. チェックリスト自動生成

#### 手順からチェック項目を抽出
```python
def extract_checklist_items(text):
    """チェックリスト項目を抽出"""
    checklist_patterns = [
        r'確認[：:]\s*(.+)',
        r'チェック[：:]\s*(.+)',
        r'✓\s*(.+)',
        r'☐\s*(.+)',
        r'□\s*(.+)',
    ]
    
    items = []
    for pattern in checklist_patterns:
        matches = re.finditer(pattern, text)
        items.extend([m.group(1).strip() for m in matches])
    
    return items
```

## 🔧 実装プラン

### Phase 1: メタデータ強化
```python
class ProcedureDocumentParser:
    """作業手順書専用パーサー"""
    
    def parse(self, pdf_path):
        """PDFを解析して構造化"""
        pages = extract_text_from_pdf(pdf_path)
        
        documents = []
        for page_num, text in pages.items():
            metadata = {
                "page": page_num,
                "step_numbers": self.extract_step_numbers(text),
                "warnings": self.detect_warnings(text),
                "figure_refs": self.extract_figure_references(text),
                "checklist": self.extract_checklist_items(text),
                "section_type": self.classify_section(text)
            }
            
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)
        
        return documents
```

### Phase 2: 専用プロンプト
```python
PROCEDURE_SYSTEM_PROMPT = """
あなたは作業手順書の専門家です。以下のルールに従って回答してください：

1. 手順は番号順に説明する
2. 「図X」「表Y」を参照する場合は、画像を適切な位置に配置
3. 警告・注意事項は必ず強調表示
4. チェックリストがある場合は箇条書きで明示
5. 前提条件と事後確認を明確に分ける

回答フォーマット：
## 前提条件
- ...

## 手順
1. [ステップ1の説明]
   [画像1]
   ⚠️ 注意: ...

2. [ステップ2の説明]
   ...

## 確認事項
- [ ] ...
"""
```

### Phase 3: 検索の最適化
```python
def search_procedure_steps(query, index, start_step=None, end_step=None):
    """手順番号を指定した検索"""
    
    # メタデータフィルタ
    filters = {}
    if start_step:
        filters["step_numbers_min"] = start_step
    if end_step:
        filters["step_numbers_max"] = end_step
    
    # 検索実行
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters
    )
    
    return query_engine.query(query)
```

### Phase 4: UI改善
```python
# サイドバーに手順書専用オプション
with st.sidebar:
    st.subheader("📋 作業手順書モード")
    
    show_step_numbers = st.checkbox("ステップ番号を表示", value=True)
    highlight_warnings = st.checkbox("警告を強調表示", value=True)
    auto_checklist = st.checkbox("チェックリスト自動生成", value=True)
    
    # ステップ範囲指定
    st.subheader("🔍 検索範囲")
    step_range = st.slider("ステップ範囲", 1, 100, (1, 100))
```

## 📊 期待される改善効果

| 項目 | 現状 | 改善後 | 効果 |
|-----|------|--------|------|
| 手順の順序理解 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +67% |
| 図表との紐付け | ⭐⭐ | ⭐⭐⭐⭐⭐ | +150% |
| 注意事項の認識 | ⭐⭐ | ⭐⭐⭐⭐⭐ | +150% |
| チェックリスト生成 | ❌ | ⭐⭐⭐⭐⭐ | 新機能 |

## 🎯 使用例

### Before（現状）
```
Q: ステップ3の注意事項は？

A: ドキュメントによると、設定を確認する必要があります。
[画像がランダムに表示]
```

### After（改善後）
```
Q: ステップ3の注意事項は？

A: ## ステップ3: データベース接続設定

⚠️ **重要な注意事項:**
1. 接続前に必ずバックアップを取得してください
2. 本番環境では読み取り専用モードで接続

[図3: 接続設定画面のスクリーンショット]

### 確認チェックリスト
- [ ] バックアップ取得済み
- [ ] 接続文字列の確認
- [ ] 権限設定の確認
```

## 🚀 実装の優先順位

### 優先度:高（すぐ実装）
1. ✅ 手順番号の抽出
2. ✅ 警告キーワードの検出
3. ✅ 図表参照の解析

### 優先度:中（次フェーズ）
4. ⬜ 専用プロンプト適用
5. ⬜ チェックリスト自動生成
6. ⬜ UI改善

### 優先度:低（将来的に）
7. ⬜ 手順の依存関係分析
8. ⬜ 作業時間の推定
9. ⬜ 複数手順書の統合

## 💡 追加アイデア

### 1. 作業履歴の記録
```python
# 実施した手順を記録
completed_steps = st.session_state.get("completed_steps", [])

if st.button("このステップを完了"):
    completed_steps.append({
        "step": current_step,
        "timestamp": datetime.now(),
        "user": st.session_state.user_id
    })
```

### 2. エラー予測
```python
# よくある失敗パターンを学習
common_errors = {
    "ステップ3": [
        "接続タイムアウト → ファイアウォール設定を確認",
        "認証エラー → パスワードの有効期限を確認"
    ]
}
```

### 3. 動画との連携
```python
# 手順に対応する動画を紐付け
video_links = {
    "ステップ1": "https://youtu.be/xxx",
    "ステップ2": "https://youtu.be/yyy"
}
```

---

**実装しますか？どの機能から始めましょうか？**
