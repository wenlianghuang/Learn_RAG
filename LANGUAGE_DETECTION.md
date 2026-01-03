# 語言自動檢測功能說明

## 📋 功能概述

系統現在支援自動檢測輸入語言，並根據檢測結果使用相應的語言回答：

- **中文問題** → **中文回答**
- **英文問題** → **英文回答**

## 🔍 工作原理

### 語言檢測算法

系統使用簡單但有效的語言檢測方法：

1. **中文字符檢測**：檢查文本中是否包含中文字符（CJK 統一表意文字範圍）
2. **比例計算**：計算中文字符在總字符中的比例
3. **語言判斷**：
   - 如果中文字符比例 > 20% → 判定為中文
   - 否則 → 判定為英文

### 檢測範例

```python
from src import PromptFormatter

formatter = PromptFormatter()

# 中文檢測
formatter.detect_language("這是一個測試問題")  # 返回 "zh"

# 英文檢測
formatter.detect_language("This is a test question")  # 返回 "en"

# 混合文本（主要語言）
formatter.detect_language("What is 機器學習?")  # 返回 "en"（英文為主）
formatter.detect_language("機器學習是什麼？What is it?")  # 返回 "zh"（中文為主）
```

## 🎯 使用方式

### 自動模式（預設）

系統預設啟用自動語言檢測：

```python
from src import PromptFormatter, RAGPipeline, OllamaLLM

# 初始化（自動檢測語言）
formatter = PromptFormatter(auto_detect_language=True)

# 中文查詢 → 中文回答
query_zh = "這份文檔的主要內容是什麼？"
prompt_zh = formatter.create_prompt(query_zh, context)
answer_zh = llm.generate(prompt_zh)  # LLM 會用中文回答

# 英文查詢 → 英文回答
query_en = "What is the main content of this document?"
prompt_en = formatter.create_prompt(query_en, context)
answer_en = llm.generate(prompt_en)  # LLM 會用英文回答
```

### 手動指定語言

如果需要禁用自動檢測，可以手動指定：

```python
# 禁用自動檢測，始終使用中文
formatter = PromptFormatter(auto_detect_language=False)

# 或手動指定系統提示詞
system_prompt_zh = formatter.get_system_prompt("zh")
system_prompt_en = formatter.get_system_prompt("en")
```

## 📝 系統提示詞

### 中文提示詞

當檢測到中文時，系統會使用以下提示詞：

```
你是一個專業的 AI 研究助手，專門回答關於機器學習、
深度學習和自然語言處理的問題。

請基於以下提供的學術論文片段來回答用戶的問題。
每個片段都標註了來源論文的資訊。

回答要求：
1. 基於提供的上下文回答問題
2. 如果上下文不足以回答，請明確說明
3. 在回答中引用具體的論文來源（使用 arXiv ID）
4. 如果不同論文有不同觀點，請分別說明
5. 保持回答簡潔、準確、專業
6. **重要：請使用與用戶問題相同的語言回答**
```

### 英文提示詞

當檢測到英文時，系統會使用以下提示詞：

```
You are a professional AI research assistant specializing in 
machine learning, deep learning, and natural language processing.

Please answer the user's question based on the provided academic paper excerpts. 
Each excerpt is labeled with source paper information.

Answer requirements:
1. Answer the question based on the provided context
2. If the context is insufficient, clearly state so
3. Cite specific paper sources in your answer (using arXiv ID)
4. If different papers have different viewpoints, explain them separately
5. Keep answers concise, accurate, and professional
6. **Important: Please answer in the same language as the user's question**
```

## 🔧 技術實現

### 核心方法

1. **`detect_language(text: str) -> str`**
   - 靜態方法，檢測文本的主要語言
   - 返回 "zh" 或 "en"

2. **`get_system_prompt(language: str) -> str`**
   - 根據語言代碼獲取對應的系統提示詞
   - 支援 "zh" 和 "en"

3. **`create_prompt(query, context, system_prompt=None)`**
   - 自動檢測查詢語言
   - 選擇相應的系統提示詞和格式
   - 生成完整的 prompt

### 代碼位置

- `src/prompt_formatter.py` - 語言檢測和提示詞生成
- `main.py` - 對比測試中的語言檢測應用

## 💡 使用建議

### ✅ 最佳實踐

1. **保持問題語言一致**：
   - 如果問題是中文，整個問題都用中文
   - 如果問題是英文，整個問題都用英文

2. **避免混合語言**：
   - 避免在同一個問題中混合中英文
   - 如果必須混合，系統會根據主要語言判斷

3. **測試不同語言**：
   - 可以分別測試中文和英文問題
   - 驗證系統是否正確切換語言

### ⚠️ 注意事項

1. **混合語言問題**：
   - 如果問題包含中英文混合，系統會根據主要語言判斷
   - 例如："What is 機器學習?" 可能被判定為英文

2. **短文本檢測**：
   - 對於非常短的文本（< 5 個字符），檢測可能不夠準確
   - 建議使用完整的問題句子

3. **LLM 模型支持**：
   - 確保使用的 LLM 模型支援多語言
   - 大多數現代模型（如 llama3.2, deepseek-r1）都支援中英文

## 🧪 測試範例

### 測試腳本

```python
from src import PromptFormatter, OllamaLLM, RAGPipeline

# 初始化
formatter = PromptFormatter()
llm = OllamaLLM(model_name="deepseek-r1:7b")
# ... 初始化 RAG pipeline ...

# 測試中文
query_zh = "這份報告的主要結論是什麼？"
results_zh = rag_pipeline.query(query_zh, top_k=3)
context_zh = formatter.format_context(results_zh)
prompt_zh = formatter.create_prompt(query_zh, context_zh)
answer_zh = llm.generate(prompt_zh)
print("中文回答:", answer_zh)

# 測試英文
query_en = "What are the main conclusions of this report?"
results_en = rag_pipeline.query(query_en, top_k=3)
context_en = formatter.format_context(results_en)
prompt_en = formatter.create_prompt(query_en, context_en)
answer_en = llm.generate(prompt_en)
print("English Answer:", answer_en)
```

## 📊 預期效果

### 中文問題範例

**輸入**：
```
這份文檔中提到的主要技術是什麼？
```

**輸出**（中文）：
```
根據文檔內容，主要技術包括：
1. Transformer 架構...
2. 注意力機制...
[引用來源：arXiv:xxxx.xxxxx]
```

### 英文問題範例

**輸入**：
```
What are the main technologies mentioned in this document?
```

**輸出**（英文）：
```
According to the document, the main technologies include:
1. Transformer architecture...
2. Attention mechanism...
[Source: arXiv:xxxx.xxxxx]
```

## 🎉 總結

語言自動檢測功能讓系統能夠：
- ✅ 自動識別用戶問題的語言
- ✅ 使用相應的語言回答
- ✅ 提供更好的用戶體驗
- ✅ 無需手動切換語言設置

現在你可以用中文或英文提問，系統會自動用相同的語言回答！

