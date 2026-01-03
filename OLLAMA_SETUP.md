# Ollama 設置指南（16GB MacBook Air）

## 📋 概述

本指南將幫助你在 16GB MacBook Air 上設置 Ollama，以便運行本地 LLM 來生成 RAG 回答。

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

### 3. 下載推薦的模型

對於 16GB MacBook Air，推薦以下模型（按優先順序）：

#### 🥇 最佳選擇：llama3.2:3b（推薦）
```bash
ollama pull llama3.2:3b
```
- **內存需求**: ~4GB
- **質量**: 良好
- **速度**: 快速
- **適合**: 大多數 RAG 任務

#### 🥈 輕量級選擇：llama3.2:1b
```bash
ollama pull llama3.2:1b
```
- **內存需求**: ~2GB
- **質量**: 基礎
- **速度**: 極快
- **適合**: 簡單問答

#### 🥉 高質量選擇：phi3:mini
```bash
ollama pull phi3:mini
```
- **內存需求**: ~3GB
- **質量**: 良好
- **速度**: 快速
- **適合**: 需要更高質量的回答

#### 其他選擇

```bash
# Google Gemma 2B
ollama pull gemma:2b

# Mistral 7B（如果內存足夠）
ollama pull mistral:7b
```

### 4. 驗證安裝

```bash
# 檢查 Ollama 是否運行
ollama list

# 測試模型
ollama run llama3.2:3b "Hello, how are you?"
```

## 🔧 在 RAG 系統中使用

### 基本使用

```python
from src import OllamaLLM, PromptFormatter, RAGPipeline

# 初始化 LLM
llm = OllamaLLM(model_name="llama3.2:3b")

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

### 查看推薦模型

```python
from src import OllamaLLM

# 顯示所有推薦的模型
OllamaLLM.print_recommended_models()
```

## ⚙️ 性能優化建議

### 1. 關閉不必要的應用
運行 LLM 時，關閉其他佔用內存的應用程序。

### 2. 調整模型大小
- 如果內存不足，使用 `llama3.2:1b`
- 如果內存充足，可以嘗試 `llama3.2:3b` 或 `phi3:mini`

### 3. 調整生成參數
```python
# 較快的生成（較少 token）
answer = llm.generate(prompt, max_tokens=300)

# 較慢但更完整的生成
answer = llm.generate(prompt, max_tokens=1000, temperature=0.8)
```

### 4. 使用流式輸出
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
ollama pull llama3.2:3b
```

### Q: 內存不足
**A**: 
1. 使用更小的模型（如 `llama3.2:1b`）
2. 關閉其他應用程序
3. 減少 `max_tokens` 參數

### Q: 生成速度慢
**A**: 
1. 使用更小的模型
2. 減少 `max_tokens`
3. 確保 MacBook Air 有足夠的散熱

## 📊 模型對比

| 模型 | 內存需求 | 質量 | 速度 | 推薦場景 |
|------|---------|------|------|---------|
| llama3.2:1b | ~2GB | 基礎 | ⚡⚡⚡ | 快速測試 |
| llama3.2:3b | ~4GB | 良好 | ⚡⚡ | **推薦** |
| phi3:mini | ~3GB | 良好 | ⚡⚡ | 高質量需求 |
| gemma:2b | ~3GB | 良好 | ⚡⚡ | 替代選擇 |
| mistral:7b | ~8GB | 優秀 | ⚡ | 內存充足時 |

## 🎯 最佳實踐

1. **首次使用**: 從 `llama3.2:3b` 開始
2. **測試階段**: 使用 `llama3.2:1b` 快速迭代
3. **生產使用**: 根據內存情況選擇 `llama3.2:3b` 或 `phi3:mini`
4. **格式化風格**: 
   - 詳細回答用 `detailed`
   - 快速測試用 `simple`
   - 節省 token 用 `minimal`

## 📚 更多資源

- [Ollama 官方文檔](https://github.com/ollama/ollama)
- [模型列表](https://ollama.com/library)
- [性能優化指南](https://github.com/ollama/ollama/blob/main/docs/performance.md)

