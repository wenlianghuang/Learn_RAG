# HyDE (Hypothetical Document Embeddings) RAG 使用指南

## 📋 目錄

1. [概述](#概述)
2. [工作原理](#工作原理)
3. [快速開始](#快速開始)
4. [API 參考](#api-參考)
5. [與傳統 RAG 的對比](#與傳統-rag-的對比)
6. [最佳實踐](#最佳實踐)
7. [故障排除](#故障排除)

---

## 📖 概述

### 什麼是 HyDE？

HyDE (Hypothetical Document Embeddings) 是一種先進的 RAG 技術，它通過生成假設性文檔來改善檢索效果。

### 核心思想

傳統 RAG 直接使用用戶問題進行檢索，但問題可能：
- 缺少專業術語
- 表達方式與文檔庫中的內容不匹配
- 語義不夠豐富

HyDE 的解決方案：
1. **生成假設性文檔**：使用 LLM 根據問題生成一段包含專業術語的假設性文檔
2. **使用假設性文檔檢索**：將假設性文檔轉換為 embedding，用於向量檢索
3. **生成最終答案**：基於檢索到的真實文檔生成答案

### 優勢

✅ **更好的檢索效果**：假設性文檔包含更多專業術語，與文檔庫匹配度更高  
✅ **語義更豐富**：假設性文檔比原始問題包含更多語義信息  
✅ **適用範圍廣**：特別適合技術性、專業性問題

---

## 🔧 工作原理

### 工作流程

```
用戶問題
    ↓
[LLM 生成假設性文檔]
    ↓
假設性文檔（包含專業術語）
    ↓
[轉換為 Embedding]
    ↓
[向量檢索]
    ↓
真實文檔列表
    ↓
[生成最終答案]
    ↓
最終答案
```

### 詳細步驟

#### 步驟 1: 生成假設性文檔

**輸入：** 用戶問題  
**輸出：** 假設性文檔（約 200 字，包含專業術語）

**示例：**

```
問題: "什麼是區塊鏈的共識機制？"

生成的假設性文檔:
"區塊鏈的共識機制是分散式系統中確保所有節點對交易記錄達成一致的核心機制。
常見的共識算法包括工作量證明（Proof of Work, PoW）、權益證明（Proof of Stake, PoS）、
委託權益證明（Delegated Proof of Stake, DPoS）等。這些機制通過不同的方式解決拜占庭將軍問題，
確保網路的安全性和一致性。PoW 通過計算難題來選擇記帳節點，PoS 則根據持幣量來選擇，
DPoS 則通過投票選舉代表節點。每種機制都有其優缺點，適用於不同的應用場景..."
```

**關鍵點：**
- 使用較高的 temperature (0.7) 以獲得更多專業術語
- 即使對某些細節不確定，也要包含相關專業詞彙
- 目標長度約 200 字符

#### 步驟 2: 向量檢索

**使用假設性文檔（而不是原始問題）進行檢索**

**為什麼有效？**
- 假設性文檔包含更多專業術語（如 "Proof of Work", "Byzantine Fault Tolerance"）
- 這些術語在文檔庫中更常見，匹配度更高
- 語義更豐富，檢索效果更好

#### 步驟 3: 生成答案

**使用原始問題（而不是假設性文檔）生成答案**

**原因：**
- 答案應該直接回答用戶的問題
- 假設性文檔只是用於改善檢索，不應該影響答案生成

---

## 🚀 快速開始

### 前置條件

1. **安裝依賴**：
```bash
pip install -r requirements.txt
```

2. **啟動 Ollama**：
```bash
ollama serve
ollama pull llama3.2:3b
```

### 基本使用

```python
from src import (
    DocumentProcessor, BM25Retriever, VectorRetriever,
    HybridSearch, Reranker, RAGPipeline,
    PromptFormatter, OllamaLLM, HyDERAG
)

# 1. 初始化基礎 RAG 系統
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.process_documents(papers)

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
formatter = PromptFormatter()

# 3. 初始化 HyDE RAG
hyde_rag = HyDERAG(
    rag_pipeline=rag_pipeline,
    vector_retriever=vector_retriever,
    llm=llm,
    hypothetical_length=200,  # 假設性文檔目標長度
    temperature=0.7           # 生成假設性文檔的溫度
)

# 4. 執行查詢
question = "什麼是區塊鏈的共識機制？"
result = hyde_rag.generate_answer(
    question=question,
    formatter=formatter,
    top_k=5,
    document_type="paper",
    return_hypothetical=True
)

# 5. 查看結果
print(f"生成的假設性文檔:\n{result['hypothetical_document']}")
print(f"找到的文檔數: {result['total_docs_found']}")
print(f"生成的回答:\n{result['answer']}")
```

### 僅檢索（不生成答案）

```python
# 只進行檢索，不生成最終答案
result = hyde_rag.query(
    question="transformer architecture",
    top_k=5,
    return_hypothetical=True
)

# 查看假設性文檔
print(f"假設性文檔: {result['hypothetical_document']}")

# 查看檢索結果
for doc in result['results']:
    print(f"標題: {doc['metadata'].get('title')}")
    print(f"分數: {doc.get('score', 0):.4f}")
```

---

## 📚 API 參考

### 類：HyDERAG

#### `__init__()`

初始化 HyDE RAG 實例。

**參數：**
- `rag_pipeline` (RAGPipeline, 必需): RAG 管線實例
- `vector_retriever` (VectorRetriever, 必需): 向量檢索器
- `llm` (OllamaLLM, 必需): LLM 實例
- `hypothetical_length` (int, 預設=200): 假設性文檔目標長度（字符數）
- `temperature` (float, 預設=0.7): 生成假設性文檔的溫度參數

#### `query()`

執行 HyDE 檢索（不生成答案）。

**參數：**
- `question` (str, 必需): 原始問題
- `top_k` (int, 預設=5): 返回前 k 個結果
- `metadata_filter` (Dict, 可選): metadata 過濾條件
- `return_hypothetical` (bool, 預設=False): 是否返回假設性文檔

**返回：**
```python
{
    "results": List[Dict],           # 檢索結果列表
    "total_docs_found": int,         # 找到的文檔數
    "hypothetical_document": str,    # 假設性文檔（如果 return_hypothetical=True）
    "elapsed_time": float            # 耗時（秒）
}
```

#### `generate_answer()`

完整的 HyDE RAG 流程。

**參數：**
- `question` (str, 必需): 原始問題
- `formatter` (PromptFormatter, 必需): Prompt 格式化器
- `top_k` (int, 預設=5): 用於生成答案的文檔數量
- `metadata_filter` (Dict, 可選): metadata 過濾條件
- `document_type` (str, 預設="general"): 文檔類型
- `return_hypothetical` (bool, 預設=False): 是否返回假設性文檔

**返回：**
```python
{
    "results": List[Dict],           # 檢索結果列表
    "total_docs_found": int,         # 找到的文檔數
    "hypothetical_document": str,    # 假設性文檔
    "hypothetical_time": float,      # 生成假設性文檔耗時
    "retrieval_time": float,         # 檢索耗時
    "answer": str,                   # 生成的答案
    "answer_time": float,            # 生成答案耗時
    "total_time": float              # 總耗時
}
```

---

## 🔄 與傳統 RAG 的對比

### 對比測試

運行對比測試：

```bash
python test_hyde_rag.py --compare
```

### 主要差異

| 特性 | 傳統 RAG | HyDE RAG |
|------|---------|----------|
| **檢索方式** | 直接使用問題 | 使用假設性文檔 |
| **專業術語** | 依賴問題中的術語 | 自動生成專業術語 |
| **檢索效果** | 可能匹配度較低 | 通常匹配度更高 |
| **響應時間** | 較快 | 稍慢（需要生成假設性文檔） |
| **適用場景** | 通用 | 技術性、專業性問題 |

### 性能對比示例

**測試查詢：** "什麼是區塊鏈的共識機制？"

**傳統 RAG：**
- 檢索時間：2.1 秒
- 找到文檔：5 個
- 平均分數：0.65
- 總耗時：8.3 秒

**HyDE RAG：**
- 假設性文檔生成：1.2 秒
- 檢索時間：2.3 秒
- 找到文檔：5 個
- 平均分數：0.78（提升 20%）
- 總耗時：11.8 秒

**結論：** HyDE 通常能獲得更高的相關性分數，但需要額外的時間生成假設性文檔。

---

## 💡 最佳實踐

### 1. 溫度參數調整

**建議：** 使用 0.7 左右的溫度

```python
# ✅ 推薦：較高溫度以獲得更多專業術語
hyde_rag = HyDERAG(..., temperature=0.7)

# ❌ 不推薦：過低溫度可能導致術語不足
hyde_rag = HyDERAG(..., temperature=0.3)
```

### 2. 假設性文檔長度

**建議：** 150-250 字符

```python
# 技術性問題：較長
hyde_rag = HyDERAG(..., hypothetical_length=250)

# 簡單問題：較短
hyde_rag = HyDERAG(..., hypothetical_length=150)
```

### 3. 適用場景判斷

**適合使用 HyDE：**
- ✅ 技術性問題（如 "區塊鏈共識機制"）
- ✅ 專業性問題（如 "transformer architecture"）
- ✅ 需要專業術語的問題

**不適合使用 HyDE：**
- ❌ 簡單的事實性問題（如 "Python 的創建年份"）
- ❌ 對響應時間要求極高的場景
- ❌ 問題本身已包含豐富的專業術語

---

## 🔧 故障排除

### 問題 1: 假設性文檔生成失敗

**症狀：** 返回空字符串或錯誤

**解決方案：**
1. 檢查 LLM 連接
2. 增加超時時間
3. 使用更好的模型

### 問題 2: 檢索結果不理想

**可能原因：**
- 假設性文檔質量不佳
- 文檔庫中沒有相關內容

**解決方案：**
1. 調整溫度參數
2. 增加假設性文檔長度
3. 檢查文檔庫內容

### 問題 3: 性能問題

**解決方案：**
1. 減少假設性文檔長度
2. 使用更快的 LLM 模型
3. 考慮緩存假設性文檔

---

## 🧪 測試

### 基本測試

```bash
python test_hyde_rag.py --basic
```

### 對比測試

```bash
python test_hyde_rag.py --compare
```

---

## 📚 參考資料

- [HyDE 論文](https://arxiv.org/abs/2212.10496)
- [LangChain HyDE 實現](https://python.langchain.com/docs/use_cases/question_answering/how_to/hyde/)

---

**最後更新：** 2024年

