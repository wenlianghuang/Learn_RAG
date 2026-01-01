"""
Hybrid Search 主程式：整合 BM25 和向量檢索
"""
import os
import json
from src import DocumentProcessor, BM25Retriever, VectorRetriever, HybridSearch


def main():
    """主程式：示範 hybrid search 的使用"""
    
    print("=" * 60)
    print("Hybrid Search 系統初始化中...")
    print("使用 Hugging Face embedding 模型（完全免費，本地運行）")
    print("=" * 60)
    
    # 1. 初始化文檔處理器
    print("\n[1/5] 初始化文檔處理器...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # 2. 獲取 arXiv 論文
    print("\n[2/5] 從 arXiv 獲取論文...")
    papers = processor.fetch_papers(query="cat:cs.AI", max_results=5)
    print(f"獲取了 {len(papers)} 篇論文")
    
    # 3. 處理文檔（分割成 chunks）
    print("\n[3/5] 處理文檔並分割成 chunks...")
    documents = processor.process_documents(papers)
    print(f"總共創建了 {len(documents)} 個文檔 chunks")
    
    # 顯示一些範例文檔
    if documents:
        print("\n範例文檔（第一個 chunk）：")
        print(f"標題: {documents[0]['metadata']['title']}")
        print(f"內容預覽: {documents[0]['content'][:200]}...")
    
    # 4. 初始化檢索器
    print("\n[4/5] 初始化檢索器...")
    
    # BM25 檢索器
    print("  - 初始化 BM25 檢索器...")
    bm25_retriever = BM25Retriever(documents)
    
    # 向量檢索器（使用免費的 Hugging Face 模型）
    print("  - 初始化向量檢索器（使用免費的 Hugging Face 模型）...")
    print("    首次使用時會下載模型，請稍候...")
    
    # 設置 Hugging Face 模型緩存目錄（可選：外接硬碟路徑）
    # 例如："/Volumes/ExternalDrive/huggingface_cache" (macOS)
    # 或："/mnt/external_drive/huggingface_cache" (Linux)
    # 如果為 None，則使用默認位置 ~/.cache/huggingface/
    hf_cache_dir = os.getenv("HF_CACHE_DIR", None)  # 可以通過環境變數設置
    # 或者直接在這裡指定路徑，例如：
    # hf_cache_dir = "/Volumes/YourExternalDrive/huggingface_cache"
    
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Hugging Face 模型
        persist_directory="./chroma_db",
        hf_cache_dir=hf_cache_dir  # 指定模型緩存目錄（外接硬碟路徑）
    )
    
    # 5. 初始化 Hybrid Search
    print("\n[5/5] 初始化 Hybrid Search...")
    hybrid_search = HybridSearch(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        bm25_weight=0.4,  # BM25 權重 40%
        vector_weight=0.6  # 向量檢索權重 60%
    )
    
    print("\n" + "=" * 60)
    print("系統初始化完成！")
    print("=" * 60)
    
    # 6. 執行搜尋測試
    print("\n開始搜尋測試...")
    print("-" * 60)
    
    test_queries = [
        "machine learning",
        "neural networks",
        "natural language processing",
    ]
    
    for query in test_queries:
        print(f"\n查詢: '{query}'")
        print("-" * 60)
        
        # 使用 Hybrid Search
        results = hybrid_search.search(query, top_k=3)
        
        print(f"\n找到 {len(results)} 個相關結果：\n")
        
        for i, result in enumerate(results, 1):
            print(f"結果 {i}:")
            print(f"  標題: {result['metadata']['title']}")
            print(f"  arXiv ID: {result['metadata']['arxiv_id']}")
            print(f"  混合分數: {result['hybrid_score']:.4f}")
            print(f"    - BM25 分數: {result['bm25_score']:.4f}")
            print(f"    - 向量分數: {result['vector_score']:.4f}")
            print(f"  內容預覽: {result['content'][:150]}...")
            print()
    
    # 7. 比較不同檢索方法
    print("\n" + "=" * 60)
    print("比較不同檢索方法")
    print("=" * 60)
    
    comparison_query = "deep learning"
    print(f"\n查詢: '{comparison_query}'")
    print("-" * 60)
    
    # BM25 結果
    print("\n[BM25 檢索結果]")
    bm25_results = bm25_retriever.retrieve(comparison_query, top_k=3)
    for i, result in enumerate(bm25_results, 1):
        print(f"{i}. {result['metadata']['title']} (分數: {result['score']:.4f})")
    
    # 向量檢索結果
    print("\n[向量檢索結果]")
    print("注意: 向量檢索的分數是距離分數（越小越相似）")
    vector_results = vector_retriever.retrieve(comparison_query, top_k=3)
    for i, result in enumerate(vector_results, 1):
        score = result.get('score')
        if score is not None:
            print(f"{i}. {result['metadata']['title']} (距離分數: {score:.4f})")
        else:
            print(f"{i}. {result['metadata']['title']} (分數: N/A)")
    
    # Hybrid Search 結果
    print("\n[Hybrid Search 結果]")
    hybrid_results = hybrid_search.search(comparison_query, top_k=3)
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. {result['metadata']['title']} (混合分數: {result['hybrid_score']:.4f})")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

