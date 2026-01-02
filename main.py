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
    print("  搜尋 AI、機器學習和自然語言處理相關論文...")
    papers = processor.fetch_papers(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",  # AI + 機器學習 + 自然語言處理
        max_results=40  # 增加到 40 篇以獲得更多樣化的結果
    )
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
    
    # 稀疏檢索器 (BM25)
    print("  - 初始化稀疏檢索器 (BM25)...")
    bm25_retriever = BM25Retriever(documents)
    
    # 密集檢索器 (向量)
    print("  - 初始化密集檢索器（使用免費的 Hugging Face 模型）...")
    
    # 設置 Hugging Face 模型緩存目錄（可選：外接硬碟路徑）
    hf_cache_dir = os.getenv("HF_CACHE_DIR", None)
    
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db",
        hf_cache_dir=hf_cache_dir
    )
    
    # 5. 初始化 Hybrid Search
    print("\n[5/5] 初始化 Hybrid Search...")
    print("  使用 RRF (Reciprocal Rank Fusion) 方法（預設）")
    print("  RRF 不需要分數正規化，對不同分數分佈更魯棒")
    hybrid_search = HybridSearch(
        sparse_retriever=bm25_retriever,
        dense_retriever=vector_retriever,
        fusion_method="rrf",  # 使用 RRF 方法（預設）
        rrf_k=60,  # RRF 常數 k，通常設為 60
        # 如果使用 weighted_sum 方法，可以設置權重：
        # sparse_weight=0.4,
        # dense_weight=0.6,
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
        results = hybrid_search.retrieve(query, top_k=3)
        
        print(f"\n找到 {len(results)} 個相關結果：\n")
        
        for i, result in enumerate(results, 1):
            print(f"結果 {i}:")
            print(f"  標題: {result['metadata']['title']}")
            print(f"  arXiv ID: {result['metadata']['arxiv_id']}")
            print(f"  混合分數: {result['hybrid_score']:.4f}")
            
            # 根據融合方法顯示不同的資訊
            if 'rrf_score' in result:
                # RRF 方法
                print(f"    - RRF 分數: {result['rrf_score']:.4f}")
                if result.get('sparse_rank') is not None:
                    print(f"    - BM25 排名: {result['sparse_rank']} (分數: {result.get('sparse_score', 0.0):.4f})")
                if result.get('dense_rank') is not None:
                    print(f"    - 向量排名: {result['dense_rank']} (分數: {result.get('dense_score', 0.0):.4f})")
            else:
                # Weighted Sum 方法
                print(f"    - 稀疏(BM25)分數: {result.get('sparse_score', 0.0):.4f}")
                print(f"    - 密集(向量)分數: {result.get('dense_score', 0.0):.4f}")
            
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
    print("\n[BM25 檢索結果] (分數越高越好)")
    bm25_results = bm25_retriever.retrieve(comparison_query, top_k=3)
    for i, result in enumerate(bm25_results, 1):
        print(f"{i}. {result['metadata']['title']} (分數: {result['score']:.4f})")
    
    # 向量檢索結果
    print("\n[向量檢索結果] (分數越高越好)")
    vector_results = vector_retriever.retrieve(comparison_query, top_k=3)
    for i, result in enumerate(vector_results, 1):
        score = result.get('score')
        if score is not None:
            print(f"{i}. {result['metadata']['title']} (相似度分數: {score:.4f})")
        else:
            print(f"{i}. {result['metadata']['title']} (分數: N/A)")
    
    # Hybrid Search 結果（使用 RRF）
    print("\n[Hybrid Search 結果 (RRF)] (分數越高越好)")
    hybrid_results = hybrid_search.retrieve(comparison_query, top_k=3)
    for i, result in enumerate(hybrid_results, 1):
        if 'rrf_score' in result:
            print(f"{i}. {result['metadata']['title']} (RRF 分數: {result['rrf_score']:.4f})")
        else:
            print(f"{i}. {result['metadata']['title']} (混合分數: {result['hybrid_score']:.4f})")
    
    # 比較 RRF 和 Weighted Sum 方法
    print("\n" + "=" * 60)
    print("比較 RRF 和 Weighted Sum 融合方法")
    print("=" * 60)
    
    comparison_query2 = "transformer architecture"
    print(f"\n查詢: '{comparison_query2}'")
    print("-" * 60)
    
    # RRF 方法
    print("\n[RRF 方法結果]")
    hybrid_search_rrf = HybridSearch(
        sparse_retriever=bm25_retriever,
        dense_retriever=vector_retriever,
        fusion_method="rrf",
        rrf_k=60
    )
    rrf_results = hybrid_search_rrf.retrieve(comparison_query2, top_k=3)
    for i, result in enumerate(rrf_results, 1):
        print(f"{i}. {result['metadata']['title']}")
        print(f"   RRF 分數: {result['rrf_score']:.4f}")
        if result.get('sparse_rank'):
            print(f"   BM25 排名: {result['sparse_rank']}, 分數: {result.get('sparse_score', 0.0):.4f}")
        if result.get('dense_rank'):
            print(f"   向量排名: {result['dense_rank']}, 分數: {result.get('dense_score', 0.0):.4f}")
    
    # Weighted Sum 方法
    print("\n[Weighted Sum 方法結果]")
    hybrid_search_weighted = HybridSearch(
        sparse_retriever=bm25_retriever,
        dense_retriever=vector_retriever,
        fusion_method="weighted_sum",
        sparse_weight=0.4,
        dense_weight=0.6
    )
    weighted_results = hybrid_search_weighted.retrieve(comparison_query2, top_k=3)
    for i, result in enumerate(weighted_results, 1):
        print(f"{i}. {result['metadata']['title']}")
        print(f"   混合分數: {result['hybrid_score']:.4f}")
        print(f"   BM25 分數: {result.get('sparse_score', 0.0):.4f}")
        print(f"   向量分數: {result.get('dense_score', 0.0):.4f}")
    
    # 8. 示範 Metadata Filtering 功能
    print("\n" + "=" * 60)
    print("Metadata Filtering 功能示範")
    print("=" * 60)
    
    if documents:
        # 獲取第一個論文的 arxiv_id 作為範例
        first_paper_id = documents[0]['metadata'].get('arxiv_id', None)
        first_paper_title = documents[0]['metadata'].get('title', '')
        
        if first_paper_id:
            print(f"\n示範：只檢索特定論文 (arXiv ID: {first_paper_id})")
            print("-" * 60)
            
            # 使用 metadata_filter 只檢索特定論文的 chunks
            filtered_results = hybrid_search.retrieve(
                query="machine learning",
                top_k=5,
                metadata_filter={"arxiv_id": first_paper_id}
            )
            
            print(f"\n找到 {len(filtered_results)} 個來自該論文的結果：\n")
            for i, result in enumerate(filtered_results, 1):
                print(f"結果 {i}:")
                print(f"  標題: {result['metadata']['title']}")
                print(f"  arXiv ID: {result['metadata']['arxiv_id']}")
                print(f"  Chunk 索引: {result['metadata'].get('chunk_index', 'N/A')}")
                print(f"  混合分數: {result['hybrid_score']:.4f}")
                print(f"  內容預覽: {result['content'][:150]}...")
                print()
        
        # 示範按標題過濾（部分匹配）
        if first_paper_title:
            print(f"\n示範：按標題關鍵字過濾（包含 '{first_paper_title[:30]}...'）")
            print("-" * 60)
            
            # 提取標題的前幾個字作為過濾條件
            title_keyword = first_paper_title.split()[0] if first_paper_title else ""
            if title_keyword:
                filtered_by_title = hybrid_search.retrieve(
                    query="machine learning",
                    top_k=3,
                    metadata_filter={"title": title_keyword}
                )
                
                print(f"\n找到 {len(filtered_by_title)} 個匹配標題的結果：\n")
                for i, result in enumerate(filtered_by_title, 1):
                    print(f"{i}. {result['metadata']['title']} (混合分數: {result['hybrid_score']:.4f})")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

