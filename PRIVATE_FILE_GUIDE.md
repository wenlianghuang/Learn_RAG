# 私有檔案 RAG 使用指南

## 📋 概述

本指南說明如何使用 RAG 系統處理私有檔案（PDF, DOCX, TXT），並對比測試有 RAG 和無 RAG 的效果。

## 🚀 快速開始

### 1. 安裝依賴

首先確保已安裝所有必要的依賴：

```bash
# 如果使用 uv
uv sync

# 或使用 pip
pip install pypdf docx2txt
```

### 2. 準備你的私有檔案

支援的檔案格式：
- **PDF** (`.pdf`)
- **Word 文檔** (`.docx`, `.doc`)
- **純文字** (`.txt`)

### 3. 運行測試腳本

```bash
# 方式 1: 直接運行，然後輸入檔案路徑和問題
python test_private_file.py

# 方式 2: 在命令行提供檔案路徑
python test_private_file.py ./documents/my_document.pdf

# 方式 3: 提供檔案路徑和問題
python test_private_file.py ./documents/my_document.pdf "這份文檔的主要內容是什麼？"
```

## 📝 使用範例

### 範例 1: 處理 PDF 檔案

```python
from src import DocumentProcessor, BM25Retriever, VectorRetriever, HybridSearch
from src import Reranker, RAGPipeline, PromptFormatter, OllamaLLM

# 1. 處理檔案
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.process_file("./my_document.pdf")

# 2. 初始化檢索系統
bm25_retriever = BM25Retriever(documents)
vector_retriever = VectorRetriever(documents, persist_directory="./chroma_db_private")
hybrid_search = HybridSearch(bm25_retriever, vector_retriever)

# 3. 初始化 RAG 管線
reranker = Reranker()
rag_pipeline = RAGPipeline(hybrid_search, reranker)

# 4. 初始化 LLM 和格式化器
llm = OllamaLLM(model_name="llama3.2:3b")
formatter = PromptFormatter()

# 5. 查詢
query = "這份文檔的主要內容是什麼？"
results = rag_pipeline.query(query, top_k=3)

# 6. 格式化並生成回答
formatted_context = formatter.format_context(results)
prompt = formatter.create_prompt(query, formatted_context)
answer = llm.generate(prompt)
print(answer)
```

### 範例 2: 處理多個檔案

```python
from src import DocumentProcessor

processor = DocumentProcessor()

# 處理多個檔案
file_paths = [
    "./documents/report1.pdf",
    "./documents/report2.docx",
    "./documents/notes.txt"
]

documents = processor.process_files(file_paths)
print(f"總共處理了 {len(documents)} 個 chunks")
```

### 範例 3: 對比測試

```python
from src import OllamaLLM, RAGPipeline, PromptFormatter
from main import test_rag_vs_no_rag

# 初始化組件（假設已經設置好）
llm = OllamaLLM(model_name="llama3.2:3b")
formatter = PromptFormatter()

# 執行對比測試
test_rag_vs_no_rag(
    llm=llm,
    rag_pipeline=rag_pipeline,
    formatter=formatter,
    query="這份文檔中提到的主要結論是什麼？",
    test_file_path="./my_document.pdf"
)
```

## 🔍 對比測試說明

對比測試會執行兩個測試：

### 測試 1: 無 RAG
- LLM 直接回答問題
- **不提供任何上下文**
- 只能基於訓練數據回答
- **如果問題涉及私有文檔內容，LLM 可能無法回答或回答錯誤**

### 測試 2: 有 RAG
- 先檢索相關文檔片段
- 將片段格式化後提供給 LLM
- LLM 基於檢索結果回答
- **即使問題涉及私有文檔內容，LLM 也能正確回答**

### 預期結果

如果 RAG 系統正常工作，你應該看到：

1. **無 RAG 的回答**：
   - 可能說"我無法回答"或"我沒有相關資訊"
   - 或者給出一個通用的、不準確的回答

2. **有 RAG 的回答**：
   - 能夠基於文檔內容給出具體、準確的回答
   - 會引用文檔中的具體資訊
   - 回答更詳細、更相關

## 💡 測試問題建議

為了更好地驗證 RAG 效果，建議使用以下類型的問題：

### ✅ 好的測試問題（涉及文檔具體內容）

- "這份文檔的主要內容是什麼？"
- "文檔中提到了哪些關鍵概念？"
- "文檔的結論是什麼？"
- "文檔中提到了哪些數據或統計資訊？"
- "文檔中建議的解決方案是什麼？"

### ❌ 不好的測試問題（太通用）

- "什麼是機器學習？"（LLM 訓練數據中已有）
- "今天天氣如何？"（與文檔無關）
- "1+1 等於多少？"（常識問題）

## 🐛 常見問題

### Q: 檔案處理失敗

**A**: 檢查：
1. 檔案路徑是否正確
2. 檔案格式是否支援（PDF, DOCX, TXT）
3. 是否已安裝必要的依賴（pypdf, docx2txt）

### Q: 檢索不到相關文檔

**A**: 可能原因：
1. 查詢與文檔內容不匹配
2. 文檔尚未載入到檢索系統
3. 嘗試使用更具體的關鍵詞

### Q: LLM 回答不準確

**A**: 嘗試：
1. 增加檢索的 top_k 數量
2. 使用更詳細的格式化風格（detailed）
3. 調整 LLM 的 temperature 參數

### Q: 內存不足

**A**: 
1. 使用更小的模型（如 llama3.2:1b）
2. 減少 chunk_size
3. 關閉其他應用程序

## 📊 性能優化建議

1. **檔案大小**：
   - 大檔案（>10MB）可能需要較長處理時間
   - 考慮將大檔案分割成多個小檔案

2. **Chunk 大小**：
   - 較小的 chunk（500-800）適合詳細檢索
   - 較大的 chunk（1500-2000）適合概括性問題

3. **檢索數量**：
   - 簡單問題：top_k=3
   - 複雜問題：top_k=5-10

## 🎯 最佳實踐

1. **檔案準備**：
   - 確保檔案文字清晰可讀
   - PDF 檔案最好包含可選文字層（不是掃描圖片）

2. **問題設計**：
   - 使用具體的問題，涉及文檔內容
   - 避免太通用或與文檔無關的問題

3. **測試流程**：
   - 先測試無 RAG，確認 LLM 無法回答
   - 再測試有 RAG，確認能正確回答
   - 對比兩者差異，驗證 RAG 效果

## 📚 更多資源

- [主程序使用說明](README.md)
- [Ollama 設置指南](OLLAMA_SETUP.md)
- [LangChain 文檔加載器](https://python.langchain.com/docs/integrations/document_loaders/)

