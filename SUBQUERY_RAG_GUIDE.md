# Sub-query Decomposition RAG 完整使用指南

## 📋 目錄

1. [概述](#概述)
2. [工作原理](#工作原理)
3. [快速開始](#快速開始)
4. [詳細參數說明](#詳細參數說明)
5. [API 參考](#api-參考)
6. [使用場景與示例](#使用場景與示例)
7. [性能優化](#性能優化)
8. [最佳實踐](#最佳實踐)
9. [與正常 RAG 的對比](#與正常-rag-的對比)
10. [故障排除](#故障排除)
11. [常見問題](#常見問題)
12. [測試與驗證](#測試與驗證)

---

## 📖 概述

### 什麼是 Sub-query Decomposition RAG？

Sub-query Decomposition RAG（子問題拆解 RAG）是一種進階的 RAG（Retrieval-Augmented Generation）技術，它通過將複雜問題拆解成多個子問題來提升檢索和生成質量。

### 核心思想

當用戶提出一個複雜的、包含多個面向的問題時，單一的檢索查詢可能無法全面覆蓋所有相關資訊。Sub-query Decomposition RAG 的解決方案是：

1. **問題拆解**：使用 LLM 將原始問題拆解成多個專注於特定面向的子問題
2. **並行檢索**：對每個子問題分別進行檢索，獲取相關文檔
3. **結果合併**：將所有子問題的檢索結果合併，去除重複，保留最相關的文檔
4. **答案生成**：基於合併後的文檔生成最終答案

### 適用場景

✅ **適合使用 Sub-query Decomposition RAG 的情況：**
- 複雜的比較問題（如「比較 A 和 B 的差異、優缺點和應用場景」）
- 多面向查詢（如「transformer architecture, attention mechanism, and optimization」）
- 綜合性問題（如「京都與大阪的賞楓交通與擁擠度比較」）
- 需要從多個角度檢索資訊的問題

❌ **不適合使用的情況：**
- 簡單的單一問題（如「什麼是機器學習？」）
- 事實性查詢（如「Python 的創建年份」）
- 對響應時間要求極高的場景

---

## 🔧 工作原理

### 工作流程圖

```
原始問題
    ↓
[LLM 子問題生成]
    ↓
子問題 1 ──┐
子問題 2 ──┤
子問題 3 ──┘
    ↓
[並行/串行檢索]
    ↓
檢索結果 1 ──┐
檢索結果 2 ──┤
檢索結果 3 ──┘
    ↓
[去重與合併]
    ↓
最終文檔列表
    ↓
[答案生成]
    ↓
最終答案
```

### 詳細步驟說明

#### 步驟 1: 子問題生成

使用 LLM 將原始問題拆解成子問題：

```python
原始問題: "比較深度學習和機器學習的差異、優缺點和應用場景"

生成的子問題:
1. 深度學習和機器學習的差異是什麼？
2. 深度學習和機器學習各自的優缺點是什麼？
3. 深度學習和機器學習的應用場景有哪些？
```

**技術細節：**
- 使用 `temperature=0.3` 以獲得更穩定的結果
- 自動檢測問題語言（中文/英文）
- 自動清理編號前綴（如 "1. ", "1) "）
- 如果生成失敗，回退到使用原始問題

#### 步驟 2: 並行/串行檢索

對每個子問題進行檢索：

**並行模式（推薦）：**
- 使用 `ThreadPoolExecutor` 並行處理
- 最多 5 個並發線程
- 適合多個子問題的情況

**串行模式：**
- 順序處理每個子問題
- 適合單個子問題或調試場景

#### 步驟 3: 去重與合併

**去重策略：**
1. 優先使用 metadata 中的唯一標識：
   - `arxiv_id + chunk_index`（論文）
   - `file_path + chunk_index`（檔案）
2. 回退到內容 hash（MD5 前 16 位）

**分數保留：**
- 如果同一文檔在多個子問題的結果中出現，保留分數更高的版本
- 分數優先級：`rerank_score` > `hybrid_score` > `score`

#### 步驟 4: 排序與篩選

- 按分數從高到低排序
- 返回前 `top_k` 個結果

---

## 🚀 快速開始

### 前置條件

1. **安裝依賴**：
```bash
# 確保已安裝所有必要的依賴
pip install -r requirements.txt
# 或使用 uv
uv sync
```

2. **啟動 Ollama**：
```bash
# 確保 Ollama 正在運行
ollama serve

# 下載模型（如果還沒有）
ollama pull llama3.2:3b
```

### 基本使用示例

#### 示例 1: 完整流程（檢索 + 生成答案）

```python
from src import (
    DocumentProcessor, BM25Retriever, VectorRetriever,
    HybridSearch, Reranker, RAGPipeline,
    PromptFormatter, OllamaLLM, SubQueryDecompositionRAG
)

# 1. 初始化基礎 RAG 系統
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.process_documents(papers)  # 或 process_file()

bm25_retriever = BM25Retriever(documents)
vector_retriever = VectorRetriever(
    documents, 
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    persist_directory="./chroma_db"
)
hybrid_search = HybridSearch(bm25_retriever, vector_retriever)

reranker = Reranker()
rag_pipeline = RAGPipeline(hybrid_search, reranker)

# 2. 初始化 LLM 和格式化器
llm = OllamaLLM(model_name="llama3.2:3b", timeout=180)
formatter = PromptFormatter(
    include_metadata=True,
    format_style="detailed"
)

# 3. 初始化 Sub-query Decomposition RAG
subquery_rag = SubQueryDecompositionRAG(
    rag_pipeline=rag_pipeline,
    llm=llm,
    max_sub_queries=3,          # 最多生成 3 個子問題
    top_k_per_subquery=5,      # 每個子問題檢索 5 個結果
    enable_parallel=True        # 啟用並行處理
)

# 4. 執行查詢
question = "比較京都與大阪的賞楓交通與擁擠度"
result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    top_k=5,
    document_type="general",
    return_sub_queries=True
)

# 5. 查看結果
print(f"生成的子問題: {result['sub_queries']}")
print(f"找到的文檔數: {result['total_docs_found']}")
print(f"總耗時: {result['total_time']:.2f}s")
print(f"生成的回答:\n{result['answer']}")
```

#### 示例 2: 僅檢索（不生成答案）

```python
# 只進行檢索，不生成最終答案
result = subquery_rag.query(
    question="transformer architecture and attention mechanism",
    top_k=5,
    return_sub_queries=True
)

# 查看檢索結果
print(f"生成的子問題 ({len(result['sub_queries'])} 個):")
for i, sq in enumerate(result['sub_queries'], 1):
    print(f"  {i}. {sq}")

print(f"\n檢索到的文檔 ({len(result['results'])} 個):")
for i, doc in enumerate(result['results'], 1):
    print(f"\n  {i}. {doc['metadata'].get('title', 'N/A')}")
    print(f"     分數: {doc.get('rerank_score', doc.get('hybrid_score', 0)):.4f}")
    print(f"     內容預覽: {doc['content'][:150]}...")
```

---

## ⚙️ 詳細參數說明

### SubQueryDecompositionRAG 初始化參數

#### `rag_pipeline` (RAGPipeline, 必需)

**說明：** 現有的 RAG 管線實例，用於執行實際的檢索操作。

**要求：**
- 必須是已初始化的 `RAGPipeline` 實例
- 應該包含配置好的 `HybridSearch` 和 `Reranker`

**示例：**
```python
rag_pipeline = RAGPipeline(
    hybrid_search=hybrid_search,
    reranker=reranker,
    recall_k=25,
    adaptive_recall=True
)
```

#### `llm` (OllamaLLM, 必需)

**說明：** LLM 實例，用於生成子問題。

**要求：**
- 必須是已初始化的 `OllamaLLM` 實例
- 確保 Ollama 服務正在運行
- 建議使用至少 3B 參數以上的模型以獲得更好的子問題質量

**推薦模型：**
- `llama3.2:3b` - 平衡性能和質量（推薦）
- `llama3.2:1b` - 快速但質量較低
- `deepseek-r1:7b` - 高質量但需要更多內存

**示例：**
```python
llm = OllamaLLM(
    model_name="llama3.2:3b",
    timeout=180  # 超時時間（秒）
)
```

#### `max_sub_queries` (int, 預設=3)

**說明：** 最多生成的子問題數量。

**範圍：** 1-10（建議 2-5）

**影響：**
- **較小值（1-2）**：適合簡單問題，響應更快
- **中等值（3-4）**：適合大多數複雜問題（推薦）
- **較大值（5-10）**：適合極其複雜的問題，但會增加檢索時間

**選擇建議：**
- 簡單問題：1-2
- 中等複雜度：2-3
- 複雜問題：3-5
- 極其複雜：5-7

**示例：**
```python
# 對於簡單問題
subquery_rag = SubQueryDecompositionRAG(
    ..., max_sub_queries=2
)

# 對於複雜問題
subquery_rag = SubQueryDecompositionRAG(
    ..., max_sub_queries=5
)
```

#### `top_k_per_subquery` (int, 預設=5)

**說明：** 每個子問題檢索的結果數量。

**範圍：** 1-50（建議 3-20）

**影響：**
- **較小值（3-5）**：檢索更快，但可能遺漏相關文檔
- **中等值（5-10）**：平衡覆蓋率和性能（推薦）
- **較大值（10-20）**：更全面的覆蓋，但檢索時間更長

**選擇建議（根據文檔庫大小）：**
- 小文檔庫（<1000 chunks）：3-5
- 中文檔庫（1000-10000 chunks）：5-10
- 大文檔庫（>10000 chunks）：10-20

**注意：** 最終返回的結果數量由 `query()` 或 `generate_answer()` 的 `top_k` 參數決定。

**示例：**
```python
# 小文檔庫
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=3
)

# 大文檔庫
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=15
)
```

#### `enable_parallel` (bool, 預設=True)

**說明：** 是否並行處理子查詢。

**影響：**
- **True（並行）**：多個子問題同時檢索，總時間約等於最慢的子查詢時間
- **False（串行）**：順序處理每個子問題，總時間等於所有子查詢時間之和

**性能對比：**
- 3 個子問題，每個耗時 2 秒：
  - 並行：約 2 秒
  - 串行：約 6 秒

**建議：**
- 多個子問題（≥2）：使用並行（`True`）
- 單個子問題或調試：使用串行（`False`）

**示例：**
```python
# 生產環境（推薦）
subquery_rag = SubQueryDecompositionRAG(
    ..., enable_parallel=True
)

# 調試模式
subquery_rag = SubQueryDecompositionRAG(
    ..., enable_parallel=False
)
```

### query() 方法參數

#### `question` (str, 必需)

**說明：** 原始問題文本。

**要求：**
- 非空字符串
- 支持中文和英文
- 建議問題長度：10-500 字符

**示例：**
```python
question = "比較深度學習和機器學習的差異、優缺點和應用場景"
```

#### `top_k` (int, 預設=5)

**說明：** 返回前 k 個結果。

**範圍：** 1-50（建議 3-10）

**注意：** 這是最終返回的結果數量，可能小於 `total_docs_found`（去重後的總文檔數）。

#### `metadata_filter` (Dict, 可選)

**說明：** metadata 過濾條件，用於限制檢索範圍。

**格式：**
```python
{
    "arxiv_id": "1234.5678",  # 只檢索特定論文
    "title": "Machine Learning",  # 標題包含關鍵詞
    "file_path": "/path/to/file.pdf"  # 特定檔案
}
```

**邏輯：** 所有條件必須同時滿足（AND 邏輯）

**示例：**
```python
# 只檢索特定論文的 chunks
result = subquery_rag.query(
    question="transformer architecture",
    metadata_filter={"arxiv_id": "1706.03762"}
)
```

#### `return_sub_queries` (bool, 預設=False)

**說明：** 是否在結果中包含子問題列表。

**用途：**
- 調試：查看生成的子問題
- 分析：了解問題拆解效果
- 日誌：記錄子問題用於後續分析

### generate_answer() 方法參數

#### `question` (str, 必需)

同 `query()` 方法的 `question` 參數。

#### `formatter` (PromptFormatter, 必需)

**說明：** Prompt 格式化器，用於格式化檢索結果和創建最終 prompt。

**要求：** 必須是已初始化的 `PromptFormatter` 實例。

**示例：**
```python
formatter = PromptFormatter(
    include_metadata=True,
    format_style="detailed"
)
```

#### `top_k` (int, 預設=5)

**說明：** 用於生成答案的文檔數量。

**建議：**
- 簡單問題：3-5
- 複雜問題：5-10
- 極其複雜：10-15

**注意：** 過多的文檔可能導致 prompt 過長，影響生成質量。

#### `metadata_filter` (Dict, 可選)

同 `query()` 方法的 `metadata_filter` 參數。

#### `document_type` (str, 預設="general")

**說明：** 文檔類型，用於調整 prompt 格式。

**選項：**
- `"paper"` - 學術論文（會包含 arXiv ID、作者等資訊）
- `"cv"` - 簡歷/履歷（會包含檔案路徑等資訊）
- `"general"` - 通用文檔（預設）

**示例：**
```python
# 處理論文
result = subquery_rag.generate_answer(
    question="transformer architecture",
    formatter=formatter,
    document_type="paper"
)

# 處理簡歷
result = subquery_rag.generate_answer(
    question="這個人的工作經驗",
    formatter=formatter,
    document_type="cv"
)
```

#### `return_sub_queries` (bool, 預設=False)

同 `query()` 方法的 `return_sub_queries` 參數。

---

## 📚 API 參考

### 類：SubQueryDecompositionRAG

#### 方法：`__init__()`

初始化 Sub-query Decomposition RAG 實例。

**簽名：**
```python
def __init__(
    self,
    rag_pipeline: RAGPipeline,
    llm: OllamaLLM,
    max_sub_queries: int = 3,
    top_k_per_subquery: int = 5,
    enable_parallel: bool = True
) -> None
```

#### 方法：`query()`

執行 Sub-query Decomposition 檢索（不生成答案）。

**簽名：**
```python
def query(
    self,
    question: str,
    top_k: int = 5,
    metadata_filter: Optional[Dict] = None,
    return_sub_queries: bool = False
) -> Dict
```

**返回：**
```python
{
    "results": List[Dict],           # 檢索結果列表，每個包含：
                                     #   - content: str
                                     #   - metadata: Dict
                                     #   - rerank_score: float (如果有)
                                     #   - hybrid_score: float (如果有)
    "total_docs_found": int,         # 去重後的總文檔數
    "sub_queries": List[str],        # 子問題列表（如果 return_sub_queries=True）
    "elapsed_time": float            # 檢索耗時（秒）
}
```

**異常：**
- `ConnectionError`: LLM 連接失敗
- `TimeoutError`: 檢索超時
- `ValueError`: 參數無效

#### 方法：`generate_answer()`

執行完整的 Sub-query Decomposition RAG 流程（檢索 + 生成答案）。

**簽名：**
```python
def generate_answer(
    self,
    question: str,
    formatter: PromptFormatter,
    top_k: int = 5,
    metadata_filter: Optional[Dict] = None,
    document_type: str = "general",
    return_sub_queries: bool = False
) -> Dict
```

**返回：**
```python
{
    "results": List[Dict],           # 檢索結果列表
    "total_docs_found": int,         # 去重後的總文檔數
    "sub_queries": List[str],        # 子問題列表（如果 return_sub_queries=True）
    "elapsed_time": float,           # 檢索耗時（秒）
    "answer": str,                   # 生成的回答
    "formatted_context": str,        # 格式化後的上下文
    "answer_time": float,            # 生成答案耗時（秒）
    "total_time": float              # 總耗時（秒）
}
```

**異常：**
- `ConnectionError`: LLM 連接失敗
- `TimeoutError`: 檢索或生成超時
- `ValueError`: 參數無效

---

## 🔍 使用場景與示例

### 場景 1: 複雜比較問題

**問題特徵：** 需要比較多個實體的多個方面

**示例：**
```python
question = "比較深度學習和機器學習的差異、優缺點和應用場景"

# 預期生成的子問題：
# 1. 深度學習和機器學習的差異是什麼？
# 2. 深度學習和機器學習各自的優缺點是什麼？
# 3. 深度學習和機器學習的應用場景有哪些？

result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    top_k=5,
    document_type="paper"
)
```

**優勢：** 每個子問題專注於一個特定面向，檢索結果更精確。

### 場景 2: 多面向技術查詢

**問題特徵：** 包含多個技術概念或主題

**示例：**
```python
question = "transformer architecture, attention mechanism, and optimization techniques"

# 預期生成的子問題：
# 1. What is transformer architecture?
# 2. How does attention mechanism work?
# 3. What are the optimization techniques for transformers?

result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    top_k=5,
    document_type="paper"
)
```

**優勢：** 分別檢索每個概念，避免單一查詢可能遺漏的資訊。

### 場景 3: 綜合性旅遊查詢

**問題特徵：** 涉及多個地點、多個方面的比較

**示例：**
```python
question = "京都與大阪的賞楓交通與擁擠度比較"

# 預期生成的子問題：
# 1. 京都賞楓的交通方式有哪些？
# 2. 大阪賞楓的交通方式有哪些？
# 3. 京都賞楓時的擁擠度如何？
# 4. 大阪賞楓時的擁擠度如何？

result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    top_k=5,
    document_type="general"
)
```

**優勢：** 從不同角度檢索，提供更全面的資訊。

### 場景 4: 學術研究查詢

**問題特徵：** 需要從多個角度理解一個研究主題

**示例：**
```python
question = "How do neural networks learn, optimize, and generalize?"

# 預期生成的子問題：
# 1. How do neural networks learn?
# 2. How are neural networks optimized?
# 3. How do neural networks generalize?

result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    top_k=5,
    document_type="paper",
    metadata_filter={"arxiv_id": "1706.03762"}  # 可選：限制範圍
)
```

### 場景 5: 僅檢索場景

**適用情況：** 只需要檢索結果，不需要生成答案

**示例：**
```python
# 批量檢索多個問題
questions = [
    "transformer architecture",
    "attention mechanism",
    "optimization techniques"
]

all_results = []
for q in questions:
    result = subquery_rag.query(
        question=q,
        top_k=5,
        return_sub_queries=True
    )
    all_results.append(result)
    
    print(f"問題: {q}")
    print(f"子問題: {result['sub_queries']}")
    print(f"找到文檔: {result['total_docs_found']}")
    print()
```

---

## ⚡ 性能優化

### 1. 並行處理優化

**建議：** 對於多個子問題，始終啟用並行處理。

```python
# ✅ 推薦：並行處理
subquery_rag = SubQueryDecompositionRAG(
    ..., enable_parallel=True
)

# ❌ 不推薦：串行處理（除非調試）
subquery_rag = SubQueryDecompositionRAG(
    ..., enable_parallel=False
)
```

**性能提升：**
- 3 個子問題：約 3 倍速度提升
- 5 個子問題：約 5 倍速度提升

### 2. 子問題數量優化

**原則：** 根據問題複雜度動態調整。

```python
def get_optimal_subquery_count(question: str) -> int:
    """根據問題複雜度返回最優子問題數量"""
    # 簡單啟發式：基於問題長度和關鍵詞數量
    length = len(question.split())
    if length < 10:
        return 2
    elif length < 20:
        return 3
    else:
        return 4

question = "比較深度學習和機器學習的差異、優缺點和應用場景"
optimal_count = get_optimal_subquery_count(question)

subquery_rag = SubQueryDecompositionRAG(
    ..., max_sub_queries=optimal_count
)
```

### 3. 檢索參數優化

**根據文檔庫大小調整：**

```python
# 小文檔庫（<1000 chunks）
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=3
)

# 中文檔庫（1000-10000 chunks）
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=5
)

# 大文檔庫（>10000 chunks）
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=10
)
```

### 4. LLM 模型選擇

**性能 vs 質量權衡：**

```python
# 快速響應（適合簡單問題）
llm = OllamaLLM(model_name="llama3.2:1b", timeout=60)

# 平衡（推薦）
llm = OllamaLLM(model_name="llama3.2:3b", timeout=180)

# 高質量（適合複雜問題）
llm = OllamaLLM(model_name="deepseek-r1:7b", timeout=300)
```

### 5. 緩存策略

**對於重複查詢，可以實現緩存：**

```python
from functools import lru_cache

class CachedSubQueryRAG(SubQueryDecompositionRAG):
    @lru_cache(maxsize=100)
    def _generate_sub_queries_cached(self, question: str) -> tuple:
        """緩存的子問題生成"""
        sub_queries = self._generate_sub_queries(question)
        return tuple(sub_queries)  # tuple 才能被 lru_cache 緩存
```

---

## 💡 最佳實踐

### 1. 問題複雜度評估

**在決定是否使用 Sub-query Decomposition 之前，評估問題複雜度：**

```python
def should_use_subquery(question: str) -> bool:
    """判斷是否應該使用 Sub-query Decomposition"""
    # 簡單問題：直接使用正常 RAG
    simple_keywords = ["什麼是", "什麼", "定義", "what is", "define"]
    if any(kw in question.lower() for kw in simple_keywords):
        return False
    
    # 複雜問題：使用 Sub-query Decomposition
    complex_keywords = ["比較", "差異", "優缺點", "比較", "compare", "difference"]
    if any(kw in question.lower() for kw in complex_keywords):
        return True
    
    # 多個主題：使用 Sub-query Decomposition
    if question.count(",") >= 2 or question.count("和") >= 2:
        return True
    
    return False
```

### 2. 參數調優流程

**系統化調優：**

1. **初始設置：** 使用預設參數
2. **測試評估：** 運行測試查詢，評估結果質量
3. **調整參數：** 根據結果調整 `max_sub_queries` 和 `top_k_per_subquery`
4. **性能測試：** 測量響應時間
5. **迭代優化：** 重複步驟 2-4

### 3. 錯誤處理

**完整的錯誤處理示例：**

```python
try:
    result = subquery_rag.generate_answer(
        question=question,
        formatter=formatter,
        top_k=5
    )
except ConnectionError as e:
    print(f"❌ LLM 連接失敗: {e}")
    print("請確保 Ollama 正在運行: ollama serve")
except TimeoutError as e:
    print(f"❌ 請求超時: {e}")
    print("建議：增加 timeout 或使用更小的模型")
except Exception as e:
    print(f"❌ 未知錯誤: {e}")
    import traceback
    traceback.print_exc()
```

### 4. 日誌記錄

**啟用日誌以追蹤問題：**

```python
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 使用時會自動記錄
result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    return_sub_queries=True  # 記錄子問題
)
```

### 5. 結果驗證

**驗證檢索結果質量：**

```python
def validate_results(result: Dict) -> bool:
    """驗證檢索結果是否合理"""
    # 檢查是否有結果
    if not result.get('results'):
        print("⚠️  未找到任何結果")
        return False
    
    # 檢查子問題數量
    sub_queries = result.get('sub_queries', [])
    if len(sub_queries) < 2:
        print("⚠️  只生成了少量子問題，可能不適合使用 Sub-query Decomposition")
    
    # 檢查文檔數量
    if result['total_docs_found'] < 3:
        print("⚠️  找到的文檔較少，可能影響答案質量")
    
    # 檢查分數
    scores = [doc.get('rerank_score', doc.get('hybrid_score', 0)) 
              for doc in result['results']]
    if max(scores) < 0.5:
        print("⚠️  最高分數較低，檢索結果可能不夠相關")
    
    return True

result = subquery_rag.generate_answer(...)
if validate_results(result):
    print("✅ 結果驗證通過")
```

---

## 🔄 與正常 RAG 的對比

### 詳細對比表

| 特性 | 正常 RAG | Sub-query Decomposition RAG |
|------|---------|------------------------------|
| **適用場景** | 簡單、單一問題 | 複雜、多面向問題 |
| **查詢示例** | "什麼是機器學習？" | "比較深度學習和機器學習的差異、優缺點" |
| **檢索方式** | 單一查詢 | 多個子查詢 |
| **檢索覆蓋率** | 可能遺漏某些面向 | 更全面的覆蓋 |
| **響應時間** | 較快（~2-5秒） | 稍慢（~5-15秒，取決於子問題數） |
| **資源消耗** | 較低 | 較高（需要額外 LLM 調用） |
| **準確性（簡單問題）** | 較好 | 可能過度複雜化 |
| **準確性（複雜問題）** | 可能遺漏資訊 | 更好 |
| **實現複雜度** | 簡單 | 中等 |
| **維護成本** | 低 | 中等 |

### 性能對比示例

**測試查詢：** "transformer architecture, attention mechanism, and optimization techniques"

**正常 RAG：**
- 檢索時間：2.3 秒
- 找到文檔：5 個
- 總耗時：8.5 秒

**Sub-query Decomposition RAG：**
- 子問題生成：1.2 秒
- 檢索時間：3.1 秒（並行）
- 找到文檔：12 個（去重後 8 個）
- 總耗時：12.8 秒

**結論：** Sub-query RAG 找到更多相關文檔，但耗時更長。

### 何時使用哪種方法？

**使用正常 RAG 的情況：**
- ✅ 簡單的事實性查詢
- ✅ 單一概念的問題
- ✅ 對響應時間要求極高
- ✅ 資源受限的環境

**使用 Sub-query Decomposition RAG 的情況：**
- ✅ 複雜的比較問題
- ✅ 多面向查詢
- ✅ 需要全面覆蓋的場景
- ✅ 答案質量優先於響應時間

---

## 🔧 故障排除

### 問題 1: LLM 連接失敗

**症狀：**
```
ConnectionError: 無法連接到 Ollama 服務
```

**解決方案：**
1. 檢查 Ollama 是否運行：
```bash
ollama serve
```

2. 檢查 Ollama API 是否可訪問：
```bash
curl http://localhost:11434/api/tags
```

3. 檢查模型是否已下載：
```bash
ollama list
```

### 問題 2: 子問題生成失敗

**症狀：** 返回的 `sub_queries` 為空或只有原始問題

**可能原因：**
- LLM 模型太小或質量不佳
- 問題格式不正確
- LLM 超時

**解決方案：**
1. 使用更好的模型：
```python
llm = OllamaLLM(model_name="llama3.2:3b")  # 而不是 1b
```

2. 增加超時時間：
```python
llm = OllamaLLM(model_name="llama3.2:3b", timeout=300)
```

3. 檢查問題格式：
```python
# 確保問題不是空的
assert len(question.strip()) > 0
```

### 問題 3: 檢索結果為空

**症狀：** `total_docs_found` 為 0

**可能原因：**
- 文檔庫中沒有相關內容
- `metadata_filter` 過於嚴格
- 檢索參數設置不當

**解決方案：**
1. 檢查文檔庫：
```python
print(f"文檔庫大小: {len(documents)} chunks")
```

2. 放寬 `metadata_filter`：
```python
# 移除過濾條件測試
result = subquery_rag.query(question, top_k=5)  # 不使用 metadata_filter
```

3. 增加 `top_k_per_subquery`：
```python
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=10  # 增加檢索數量
)
```

### 問題 4: 性能問題

**症狀：** 響應時間過長

**解決方案：**
1. 啟用並行處理：
```python
subquery_rag = SubQueryDecompositionRAG(
    ..., enable_parallel=True
)
```

2. 減少子問題數量：
```python
subquery_rag = SubQueryDecompositionRAG(
    ..., max_sub_queries=2  # 減少到 2 個
)
```

3. 減少每個子查詢的 top_k：
```python
subquery_rag = SubQueryDecompositionRAG(
    ..., top_k_per_subquery=3  # 減少檢索數量
)
```

4. 使用更快的 LLM 模型：
```python
llm = OllamaLLM(model_name="llama3.2:1b")  # 更快的模型
```

### 問題 5: 答案質量不佳

**症狀：** 生成的答案不準確或不完整

**解決方案：**
1. 增加用於生成答案的文檔數量：
```python
result = subquery_rag.generate_answer(
    ..., top_k=10  # 增加文檔數量
)
```

2. 使用更好的 LLM 模型生成答案：
```python
# 可以為生成答案使用不同的 LLM
answer_llm = OllamaLLM(model_name="deepseek-r1:7b")
# 注意：這需要修改 generate_answer 方法
```

3. 調整 prompt 格式：
```python
formatter = PromptFormatter(
    format_style="detailed",  # 使用詳細格式
    include_metadata=True
)
```

---

## ❓ 常見問題

### Q1: Sub-query Decomposition 是否總是比正常 RAG 好？

**A:** 不是。對於簡單問題，正常 RAG 可能更快、更準確。Sub-query Decomposition 主要優勢在於處理複雜、多面向的問題。

### Q2: 子問題數量應該設置為多少？

**A:** 一般建議 2-5 個。太少可能無法充分拆解問題，太多會增加檢索時間。可以根據問題複雜度動態調整。

### Q3: 並行處理是否總是更快？

**A:** 是的，對於多個子問題，並行處理通常能顯著提升性能。但對於單個子問題，並行處理沒有意義。

### Q4: 如何判斷問題是否適合使用 Sub-query Decomposition？

**A:** 如果問題包含以下特徵，適合使用：
- 多個實體的比較
- 多個面向的查詢
- 多個關鍵詞或概念
- 需要全面覆蓋的場景

### Q5: 去重是如何工作的？

**A:** 系統優先使用 metadata 中的唯一標識（如 `arxiv_id + chunk_index`），如果沒有，則使用內容的 MD5 hash。如果同一文檔在多個子問題的結果中出現，保留分數更高的版本。

### Q6: 可以自定義子問題生成的 prompt 嗎？

**A:** 目前不支持直接自定義，但可以通過修改 `SubQueryDecompositionRAG` 類的 `_generate_sub_queries` 方法來實現。

### Q7: 支持哪些語言？

**A:** 目前支持中文和英文。系統會自動檢測問題語言並使用相應的 prompt。

### Q8: 如何處理超時問題？

**A:** 可以增加 LLM 的 timeout 參數：
```python
llm = OllamaLLM(model_name="llama3.2:3b", timeout=300)  # 5 分鐘
```

---

## 🧪 測試與驗證

### 運行基本測試

```bash
# 基本功能測試
python test_subquery_rag.py
```

### 運行對比測試

```bash
# 對比 Sub-query RAG 和正常 RAG
python test_subquery_rag.py --compare
```

### 測試輸出解讀

**基本測試輸出：**
- 子問題列表
- 檢索到的文檔
- 生成的答案
- 性能統計

**對比測試輸出：**
- 兩種方法的性能對比
- 檢索結果對比
- 文檔重疊分析
- 自動觀察和建議

### 自定義測試

```python
# 測試特定問題
question = "你的測試問題"
result = subquery_rag.generate_answer(
    question=question,
    formatter=formatter,
    return_sub_queries=True
)

# 驗證結果
assert result['total_docs_found'] > 0, "應該找到至少一個文檔"
assert len(result.get('sub_queries', [])) > 0, "應該生成至少一個子問題"
assert len(result['answer']) > 0, "應該生成答案"
```

---

## 📚 參考資料

### 相關文檔

- [LangChain Sub-query Decomposition](https://python.langchain.com/docs/use_cases/question_answering/how_to/decompose/)
- [RAG 最佳實踐](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [本項目 README](../README.md)

### 相關技術

- **RAG (Retrieval-Augmented Generation)**: 檢索增強生成
- **Hybrid Search**: 混合搜尋（BM25 + 向量檢索）
- **Reranking**: 重排序技術
- **Query Decomposition**: 查詢拆解技術

---

## 📝 更新日誌

### v1.0.0 (當前版本)

- 初始實現 Sub-query Decomposition RAG
- 支持並行/串行檢索
- 支持自動去重
- 支持中英文問題

---

## 🤝 貢獻

如有問題或建議，請提交 Issue 或 Pull Request。

---

**最後更新：** 2024年
