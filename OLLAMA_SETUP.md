# Ollama 設置指南

## 📋 概述

本指南將幫助你設置 Ollama，以便運行本地 LLM 來生成 RAG 回答。

## 🚀 快速開始

### 1. 安裝 Ollama

```bash
# 訪問官網下載 macOS 版本
# https://ollama.ai/download

# 或使用 Homebrew
brew install ollama
```

### 2. 啟動 Ollama 服務

安裝後，Ollama 通常會自動啟動。如果沒有，可以手動啟動：

```bash
ollama serve
```

### 3. 下載模型

```bash
ollama pull deepseek-r1:7b
```

### 4. 驗證安裝

```bash
# 檢查 Ollama 是否運行
ollama list

# 測試模型
ollama run deepseek-r1:7b "Hello, how are you?"
```

## 🔧 在 RAG 系統中使用

### 基本使用

```python
from src import OllamaLLM, PromptFormatter, RAGPipeline

# 初始化 LLM
llm = OllamaLLM(model_name="deepseek-r1:7b")

# 檢索文檔（使用你的 RAG Pipeline）
results = rag_pipeline.query("你的問題", top_k=5)

# 格式化結果
formatter = PromptFormatter(format_style="detailed")
formatted_context = formatter.format_context(results)

# 創建 prompt
prompt = formatter.create_prompt("你的問題", formatted_context)

# 生成回答
answer = llm.generate(prompt, temperature=0.7, max_tokens=500)
print(answer)
```

## ⚙️ 性能優化建議

### 1. 關閉不必要的應用
運行 LLM 時，關閉其他佔用內存的應用程序。

### 2. 調整生成參數
```python
# 較快的生成（較少 token）
answer = llm.generate(prompt, max_tokens=300)

# 較慢但更完整的生成
answer = llm.generate(prompt, max_tokens=1000, temperature=0.8)
```

### 3. 使用流式輸出
```python
# 流式輸出，可以實時看到生成過程
answer = llm.generate(prompt, stream=True)
```

## 🐛 常見問題

### Q: 連接錯誤 "無法連接到 Ollama"
**A**: 確保 Ollama 服務正在運行：
```bash
ollama serve
```

### Q: 模型未找到
**A**: 確保已下載模型：
```bash
ollama pull deepseek-r1:7b
```

### Q: 內存不足
**A**: 
1. 關閉其他應用程序
2. 減少 `max_tokens` 參數

### Q: 生成速度慢
**A**: 
1. 減少 `max_tokens`
2. 確保 MacBook Air 有足夠的散熱

## 📚 更多資源

- [Ollama 官方文檔](https://github.com/ollama/ollama)
- [模型列表](https://ollama.com/library)
