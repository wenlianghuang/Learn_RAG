"""
測試私有檔案的 RAG 效果

這個腳本演示如何：
1. 載入私有檔案（PDF, DOCX, TXT）
2. 建立 RAG 系統
3. 對比測試：有 RAG vs 無 RAG 的效果
"""
import os
import sys
from pathlib import Path
from src import (
    DocumentProcessor,
    BM25Retriever,
    VectorRetriever,
    HybridSearch,
    Reranker,
    RAGPipeline,
    PromptFormatter,
    OllamaLLM
)
from main import test_rag_vs_no_rag


def main():
    print("=" * 60)
    print("私有檔案 RAG 測試")
    print("=" * 60)
    
    # ========== 步驟 1: 準備私有檔案 ==========
    print("\n[步驟 1] 準備私有檔案")
    print("-" * 60)
    
    # 提示用戶輸入檔案路徑
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 預設示例：可以修改為你的檔案路徑
        file_path = input("\n請輸入檔案路徑（PDF, DOCX, 或 TXT）: ").strip()
        
        if not file_path:
            print("\n⚠️  未提供檔案路徑")
            print("\n使用方法：")
            print("  python test_private_file.py <檔案路徑>")
            print("\n或直接運行，然後輸入檔案路徑")
            print("\n範例：")
            print("  python test_private_file.py ./documents/my_document.pdf")
            return
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"\n❌ 檔案不存在: {file_path}")
        print("\n請確認：")
        print("  1. 檔案路徑是否正確")
        print("  2. 檔案是否存在")
        return
    
    print(f"✓ 找到檔案: {file_path}")
    print(f"  檔案類型: {file_path.suffix}")
    print(f"  檔案大小: {file_path.stat().st_size / 1024:.2f} KB")
    
    # ========== 步驟 2: 處理檔案 ==========
    print("\n[步驟 2] 處理檔案並分割成 chunks")
    print("-" * 60)
    
    try:
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        documents = processor.process_file(str(file_path))
        print(f"✓ 處理完成，創建了 {len(documents)} 個 chunks")
        
        if documents:
            print(f"\n範例 chunk（第一個）：")
            print(f"  標題: {documents[0]['metadata']['title']}")
            print(f"  內容預覽: {documents[0]['content'][:150]}...")
    except Exception as e:
        print(f"\n❌ 處理檔案失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 步驟 3: 初始化檢索系統 ==========
    print("\n[步驟 3] 初始化檢索系統")
    print("-" * 60)
    
    try:
        print("  - 初始化 BM25 檢索器...")
        bm25_retriever = BM25Retriever(documents)
        
        print("  - 初始化向量檢索器...")
        # 使用不同的資料庫目錄避免與主程序衝突
        vector_retriever = VectorRetriever(
            documents,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory="./chroma_db_private"
        )
        
        print("  - 初始化混合搜尋...")
        hybrid_search = HybridSearch(
            sparse_retriever=bm25_retriever,
            dense_retriever=vector_retriever,
            fusion_method="rrf",
            rrf_k=60
        )
        
        print("✓ 檢索系統初始化完成")
    except Exception as e:
        print(f"\n❌ 檢索系統初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 步驟 4: 初始化重排序和 RAG 管線 ==========
    print("\n[步驟 4] 初始化重排序和 RAG 管線")
    print("-" * 60)
    
    try:
        print("  - 初始化重排序器...")
        reranker = Reranker(
            model_name="BAAI/bge-reranker-base",
            batch_size=16
        )
        
        print("  - 初始化 RAG 管線...")
        rag_pipeline = RAGPipeline(
            hybrid_search=hybrid_search,
            reranker=reranker,
            recall_k=20,
            adaptive_recall=True
        )
        
        print("✓ RAG 管線初始化完成")
    except Exception as e:
        print(f"\n❌ RAG 管線初始化失敗: {e}")
        print("   這可能是因為重排序模型下載失敗")
        print("   你可以繼續使用混合搜尋（不進行重排序）")
        return
    
    # ========== 步驟 5: 初始化 LLM 和格式化器 ==========
    print("\n[步驟 5] 初始化 LLM 和格式化器")
    print("-" * 60)
    
    try:
        print("  - 初始化 Prompt 格式化器...")
        formatter = PromptFormatter(format_style="detailed")
        
        print("  - 初始化 Ollama LLM...")
        # 顯示推薦的模型
        OllamaLLM.print_recommended_models()
        
        llm = OllamaLLM(
            model_name="llama3.2:3b",  # 適合 16GB 內存
            timeout=180
        )
        
        print("✓ LLM 和格式化器初始化完成")
    except ConnectionError as e:
        print(f"\n❌ Ollama 連接失敗: {e}")
        print("\n請按照以下步驟設置 Ollama：")
        print("  1. 安裝 Ollama: https://ollama.ai/download")
        print("  2. 啟動 Ollama 服務（通常會自動啟動）")
        print("  3. 下載模型: ollama pull llama3.2:3b")
        print("  4. 重新運行此程序")
        return
    except Exception as e:
        print(f"\n❌ 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 步驟 6: 測試問題 ==========
    print("\n[步驟 6] 準備測試問題")
    print("-" * 60)
    
    # 提示用戶輸入問題
    if len(sys.argv) > 2:
        test_query = " ".join(sys.argv[2:])
    else:
        print("\n請輸入一個測試問題（應該涉及你的私有文檔內容）：")
        print("範例：")
        print("  - '這份文檔的主要內容是什麼？'")
        print("  - '文檔中提到了哪些關鍵概念？'")
        print("  - '文檔的結論是什麼？'")
        test_query = input("\n你的問題: ").strip()
    
    if not test_query:
        print("\n⚠️  未提供測試問題")
        print("使用預設問題...")
        test_query = "這份文檔的主要內容是什麼？"
    
    print(f"\n✓ 測試問題: '{test_query}'")
    
    # ========== 步驟 7: 執行對比測試 ==========
    print("\n" + "=" * 60)
    print("開始執行對比測試")
    print("=" * 60)
    
    test_rag_vs_no_rag(
        llm=llm,
        rag_pipeline=rag_pipeline,
        formatter=formatter,
        query=test_query,
        test_file_path=str(file_path)
    )
    
    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)
    print("\n💡 提示：")
    print("  - 如果無 RAG 的回答不準確或無法回答，但有 RAG 的回答正確，")
    print("    這證明了 RAG 系統的有效性！")
    print("  - 你可以嘗試不同的問題來測試系統的各種情況")


if __name__ == "__main__":
    main()

