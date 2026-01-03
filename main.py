"""
Hybrid Search 主程式：整合 BM25 和向量檢索
"""
import os
import json
from typing import List, Dict, Optional
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


def get_test_queries() -> Dict[str, List[str]]:
    """
    獲取分類的測試查詢
    
    Returns:
        包含不同類型查詢的字典
    """
    return {
        # 關鍵詞匹配測試：測試 BM25 的優勢（精確關鍵詞匹配）
        "keyword": [
            "transformer architecture",
            "attention mechanism",
            "gradient descent",
            "neural network",
        ],
        
        # 語義理解測試：測試向量檢索的優勢（同義詞和語義相似）
        "semantic": [
            "How do deep learning models learn?",
            "What is the difference between supervised and unsupervised learning?",
            "methods for improving model generalization",
            "ways to reduce overfitting in neural networks",
        ],
        
        # 複雜查詢測試：測試混合搜尋和重排序的優勢
        "complex": [
            "transformer models with attention mechanism for natural language processing",
            "optimization techniques for training large language models efficiently",
            "recent advances in few-shot learning and meta-learning",
            "comparison between BERT and GPT architectures",
        ],
        
        # 抽象概念查詢：測試語義理解深度
        "abstract": [
            "How can AI systems understand context?",
            "What makes a model interpretable?",
            "scalability challenges in machine learning",
        ],
        
        # 邊界情況測試
        "edge_cases": [
            "AI",  # 太短
            "machine learning deep learning neural networks artificial intelligence",  # 太長，關鍵詞堆砌
        ],
    }


def compare_retrieval_methods(
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    hybrid_search: HybridSearch,
    rag_pipeline: RAGPipeline,
    queries: List[str]
) -> List[Dict]:
    """
    對比不同檢索方法的效果
    
    Args:
        bm25_retriever: BM25 檢索器
        vector_retriever: 向量檢索器
        hybrid_search: 混合搜尋實例
        rag_pipeline: RAG 管線實例
        queries: 測試查詢列表
        
    Returns:
        對比結果列表
    """
    results_comparison = []
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"查詢: '{query}'")
        print(f"{'='*60}")
        
        try:
            # 1. 僅 BM25
            bm25_results = bm25_retriever.retrieve(query, top_k=5)
            
            # 2. 僅向量檢索
            vector_results = vector_retriever.retrieve(query, top_k=5)
            
            # 3. 混合搜尋（無重排序）
            hybrid_results = hybrid_search.retrieve(query, top_k=5)
            
            # 4. 完整 RAG Pipeline（有重排序）
            rag_results, stats = rag_pipeline.query(
                query, 
                top_k=5, 
                enable_rerank=True,
                return_stats=True
            )
            
            # 分析結果差異
            comparison = {
                "query": query,
                "bm25_top1": bm25_results[0]['metadata']['title'] if bm25_results else None,
                "vector_top1": vector_results[0]['metadata']['title'] if vector_results else None,
                "hybrid_top1": hybrid_results[0]['metadata']['title'] if hybrid_results else None,
                "rag_top1": rag_results[0]['metadata']['title'] if rag_results else None,
                "stats": stats
            }
            
            results_comparison.append(comparison)
            
            # 打印對比
            print("\n[BM25 最佳結果]")
            if bm25_results:
                print(f"  {bm25_results[0]['metadata']['title']}")
                print(f"  分數: {bm25_results[0].get('score', 0):.4f}")
            
            print("\n[向量檢索最佳結果]")
            if vector_results:
                print(f"  {vector_results[0]['metadata']['title']}")
                print(f"  相似度分數: {vector_results[0].get('score', 0):.4f}")
            
            print("\n[混合搜尋最佳結果]")
            if hybrid_results:
                print(f"  {hybrid_results[0]['metadata']['title']}")
                print(f"  混合分數: {hybrid_results[0].get('hybrid_score', 0):.4f}")
            
            print("\n[RAG Pipeline 最佳結果（重排序後）]")
            if rag_results:
                print(f"  {rag_results[0]['metadata']['title']}")
                print(f"  重排序分數: {rag_results[0].get('rerank_score', 0):.4f}")
                print(f"  原始混合分數: {rag_results[0].get('hybrid_score', 0):.4f}")
                print(f"  性能: {stats['total_time']:.2f}s")
                print(f"    - 召回: {stats['recall_time']:.2f}s")
                print(f"    - 重排: {stats['rerank_time']:.2f}s")
            
        except Exception as e:
            print(f"⚠️  查詢處理出錯: {e}")
            continue
    
    return results_comparison


def evaluate_retrieval_quality(
    results: List[Dict], 
    query: str, 
    expected_keywords: Optional[List[str]] = None
) -> Dict:
    """
    評估檢索質量
    
    Args:
        results: 檢索結果列表
        query: 查詢文本
        expected_keywords: 期望的關鍵詞列表（可選）
        
    Returns:
        包含評估指標的字典
    """
    metrics = {
        "diversity": 0.0,  # 結果多樣性（不同論文的數量）
        "score_distribution": {},  # 分數分佈
        "coverage": 0.0,  # 覆蓋度（是否涵蓋查詢的各個方面）
    }
    
    if not results:
        return metrics
    
    # 1. 檢查結果多樣性
    unique_titles = set(r['metadata']['title'] for r in results)
    metrics["diversity"] = len(unique_titles) / len(results)
    
    # 2. 分數分佈
    scores = []
    for r in results:
        score = r.get('rerank_score') or r.get('hybrid_score') or r.get('score', 0)
        scores.append(score)
    
    if scores:
        metrics["score_distribution"] = {
            "max": max(scores),
            "min": min(scores),
            "avg": sum(scores) / len(scores),
        }
        
        # 計算標準差
        avg = metrics["score_distribution"]["avg"]
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        metrics["score_distribution"]["std"] = variance ** 0.5
    
    # 3. 關鍵詞覆蓋（如果提供了期望關鍵詞）
    if expected_keywords:
        content_text = " ".join([r['content'].lower() for r in results])
        found_keywords = sum(1 for kw in expected_keywords if kw.lower() in content_text)
        metrics["coverage"] = found_keywords / len(expected_keywords) if expected_keywords else 0
    
    return metrics


def comprehensive_rag_test(
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    hybrid_search: HybridSearch,
    rag_pipeline: RAGPipeline
):
    """
    全面的 RAG 系統測試
    
    Args:
        bm25_retriever: BM25 檢索器
        vector_retriever: 向量檢索器
        hybrid_search: 混合搜尋實例
        rag_pipeline: RAG 管線實例
    """
    print("\n" + "=" * 60)
    print("全面的 RAG 系統功效測試")
    print("=" * 60)
    
    # 獲取測試查詢
    test_queries_dict = get_test_queries()
    
    # 1. 基礎功能測試（使用關鍵詞和語義查詢）
    print("\n" + "-" * 60)
    print("1. 基礎功能測試")
    print("-" * 60)
    
    basic_queries = test_queries_dict["keyword"][:2] + test_queries_dict["semantic"][:2]
    
    for query in basic_queries:
        print(f"\n查詢: '{query}'")
        print("-" * 40)
        
        try:
            results, stats = rag_pipeline.query(
                query, 
                top_k=5, 
                return_stats=True
            )
            
            print(f"找到 {len(results)} 個結果")
            print(f"耗時: {stats['total_time']:.2f}s (召回: {stats['recall_time']:.2f}s, 重排: {stats['rerank_time']:.2f}s)")
            
            # 評估質量
            quality = evaluate_retrieval_quality(results, query)
            print(f"結果多樣性: {quality['diversity']:.2%}")
            if quality['score_distribution']:
                print(f"分數範圍: [{quality['score_distribution']['min']:.4f}, {quality['score_distribution']['max']:.4f}]")
                print(f"平均分數: {quality['score_distribution']['avg']:.4f}")
            
            # 顯示前 3 個結果
            for i, r in enumerate(results[:3], 1):
                print(f"\n  {i}. {r['metadata']['title']}")
                print(f"     重排序分數: {r.get('rerank_score', 0):.4f}")
                print(f"     內容預覽: {r['content'][:100]}...")
                
        except Exception as e:
            print(f"⚠️  查詢處理出錯: {e}")
            continue
    
    # 2. 複雜查詢測試
    print("\n" + "-" * 60)
    print("2. 複雜查詢測試")
    print("-" * 60)
    
    complex_queries = test_queries_dict["complex"][:2]
    
    for query in complex_queries:
        print(f"\n查詢: '{query}'")
        print("-" * 40)
        
        try:
            results, stats = rag_pipeline.query(
                query, 
                top_k=5, 
                return_stats=True
            )
            
            print(f"找到 {len(results)} 個結果")
            print(f"實際召回數量: {stats['recall_k']} 筆")
            print(f"耗時: {stats['total_time']:.2f}s")
            
            for i, r in enumerate(results[:3], 1):
                print(f"\n  {i}. {r['metadata']['title']}")
                print(f"     重排序分數: {r.get('rerank_score', 0):.4f}")
                
        except Exception as e:
            print(f"⚠️  查詢處理出錯: {e}")
            continue
    
    # 3. 方法對比測試
    print("\n" + "-" * 60)
    print("3. 方法對比測試")
    print("-" * 60)
    
    comparison_queries = [
        "transformer architecture",
        "How do neural networks learn?",
        "optimization methods for deep learning"
    ]
    
    compare_retrieval_methods(
        bm25_retriever,
        vector_retriever,
        hybrid_search,
        rag_pipeline,
        comparison_queries
    )
    
    # 4. 性能統計
    print("\n" + "-" * 60)
    print("4. 累積性能統計")
    print("-" * 60)
    
    stats = rag_pipeline.get_stats()
    print(f"總查詢數: {stats['total_queries']}")
    print(f"平均召回時間: {stats['avg_recall_time']:.2f}s")
    print(f"平均重排時間: {stats['avg_rerank_time']:.2f}s")
    print(f"平均總時間: {stats['avg_total_time']:.2f}s")
    
    # 5. 最佳實踐查詢示例
    print("\n" + "-" * 60)
    print("5. 最佳實踐查詢示例")
    print("-" * 60)
    
    best_practice_queries = [
        "How do transformer models use self-attention?",
        "What are the advantages of BERT over traditional NLP models?",
        "recent techniques for fine-tuning large language models",
    ]
    
    for query in best_practice_queries:
        print(f"\n查詢: '{query}'")
        print("-" * 40)
        
        try:
            results, stats = rag_pipeline.query(
                query, 
                top_k=3, 
                return_stats=True
            )
            
            if results:
                print(f"最佳結果: {results[0]['metadata']['title']}")
                print(f"重排序分數: {results[0].get('rerank_score', 0):.4f}")
                print(f"耗時: {stats['total_time']:.2f}s")
                
        except Exception as e:
            print(f"⚠️  查詢處理出錯: {e}")
            continue


def detect_document_type(file_path: Optional[str] = None, results: Optional[List[Dict]] = None) -> str:
    """
    自動檢測文檔類型
    
    Args:
        file_path: 文件路徑（可選）
        results: 檢索結果列表（可選，用於從 metadata 推斷）
        
    Returns:
        文檔類型 ("paper", "cv", "general")
    """
    # 優先從文件路徑判斷
    if file_path:
        file_path_lower = file_path.lower()
        if any(keyword in file_path_lower for keyword in ["cv", "resume", "履歷", "簡歷"]):
            return "cv"
        elif any(keyword in file_path_lower for keyword in ["arxiv", "paper", "論文"]):
            return "paper"
    
    # 從檢索結果的 metadata 判斷
    if results:
        for result in results:
            metadata = result.get("metadata", {})
            # 如果有 arxiv_id，很可能是論文
            if "arxiv_id" in metadata:
                return "paper"
            # 如果有 file_path，檢查文件名
            if "file_path" in metadata:
                file_path_lower = str(metadata.get("file_path", "")).lower()
                if any(keyword in file_path_lower for keyword in ["cv", "resume", "履歷", "簡歷"]):
                    return "cv"
            # 檢查標題
            title = str(metadata.get("title", "")).lower()
            if any(keyword in title for keyword in ["cv", "resume", "curriculum vitae", "履歷", "簡歷"]):
                return "cv"
    
    # 預設為通用類型
    return "general"


def test_rag_vs_no_rag(
    llm: OllamaLLM,
    rag_pipeline: RAGPipeline,
    formatter: PromptFormatter,
    query: str,
    test_file_path: Optional[str] = None
):
    """
    對比測試：有 RAG vs 無 RAG
    
    這個測試可以驗證 RAG 系統的效果：
    - 無 RAG：LLM 只能基於訓練數據回答（可能無法回答私有文檔的問題）
    - 有 RAG：LLM 可以基於檢索到的私有文檔回答（應該能正確回答）
    
    Args:
        llm: LLM 實例
        rag_pipeline: RAG 管線
        formatter: Prompt 格式化器
        query: 測試問題（應該涉及私有文檔的內容）
        test_file_path: 測試文件路徑（可選，用於顯示文件信息）
    """
    print("\n" + "="*60)
    print("RAG 效果對比測試")
    print("="*60)
    print(f"\n測試問題: '{query}'")
    if test_file_path:
        print(f"測試文件: {test_file_path}")
    print("-"*60)
    
    # 檢測查詢語言
    detected_lang = PromptFormatter.detect_language(query)
    is_chinese = detected_lang == "zh"
    
    # 1. 無 RAG：直接問 LLM
    print("\n[測試 1] 無 RAG - 直接問 LLM（不提供任何上下文）")
    print("-"*60)
    try:
        if is_chinese:
            no_rag_prompt = f"""請回答以下問題：

{query}

請注意：你只能基於你的訓練數據來回答，無法訪問任何外部文檔。
請使用中文回答。"""
        else:
            no_rag_prompt = f"""Please answer the following question:

{query}

Note: You can only answer based on your training data and cannot access any external documents.
Please answer in English."""
        
        print("生成回答中...")
        no_rag_answer = llm.generate(
            prompt=no_rag_prompt,
            temperature=0.7,
            max_tokens=2048
        )
        print("\nLLM 回答（無 RAG）：")
        print("-" * 40)
        print(no_rag_answer)
        print("-" * 40)
        print("\n✅ 無 RAG 回答完成")
    except Exception as e:
        print(f"❌ 無 RAG 測試失敗: {e}")
        no_rag_answer = None
    
    # 2. 有 RAG：使用檢索增強
    print("\n" + "-"*60)
    print("[測試 2] 有 RAG - 使用檢索增強（提供相關文檔片段）")
    print("-"*60)
    
    try:
        # 檢索相關文檔
        print("檢索相關文檔中...")
        rag_results = rag_pipeline.query(
            text=query,
            top_k=3,
            enable_rerank=True
        )
        
        if not rag_results:
            print("⚠️  未找到相關文檔")
            print("這可能意味著：")
            print("  1. 查詢與文檔內容不匹配")
            print("  2. 文檔尚未載入到檢索系統")
            return
        
        print(f"✅ 找到 {len(rag_results)} 個相關片段")
        
        # 自動檢測文檔類型
        document_type = detect_document_type(test_file_path, rag_results)
        print(f"  檢測到的文檔類型: {document_type}")
        
        # 格式化並生成回答（傳入文檔類型）
        formatted_context = formatter.format_context(
            rag_results, 
            format_style="detailed",
            document_type=document_type
        )
        rag_prompt = formatter.create_prompt(
            query, 
            formatted_context,
            document_type=document_type
        )
        
        print("\n生成回答中...")
        rag_answer = llm.generate(
            prompt=rag_prompt,
            temperature=0.7,
            max_tokens=2048
        )
        
        print("\nLLM 回答（有 RAG）：")
        print("-" * 40)
        print(rag_answer)
        print("-" * 40)
        
        # 顯示檢索到的文檔片段
        print("\n" + "-"*60)
        print("檢索到的文檔片段（LLM 的參考來源）：")
        print("-"*60)
        for i, result in enumerate(rag_results, 1):
            print(f"\n片段 {i}:")
            metadata = result['metadata']
            print(f"  來源: {metadata.get('title', 'N/A')}")
            if 'file_path' in metadata:
                print(f"  文件: {metadata['file_path']}")
            elif 'arxiv_id' in metadata:
                print(f"  arXiv ID: {metadata['arxiv_id']}")
            print(f"  相關性分數: {result.get('rerank_score', result.get('hybrid_score', 0)):.4f}")
            print(f"  內容預覽: {result['content'][:200]}...")
        
        print("\n✅ 有 RAG 回答完成")
        
    except Exception as e:
        print(f"❌ 有 RAG 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        rag_answer = None
    
    # 3. 對比總結
    print("\n" + "="*60)
    print("對比總結")
    print("="*60)
    
    if no_rag_answer and rag_answer:
        print("\n✅ 兩個測試都成功完成")
        print("\n關鍵差異：")
        print("1. 無 RAG：LLM 只能基於訓練數據回答")
        print("   - 如果問題涉及私有文檔內容，LLM 可能無法回答或回答錯誤")
        print("2. 有 RAG：LLM 可以基於檢索到的私有文檔回答")
        print("   - 即使問題涉及私有文檔內容，LLM 也能基於檢索結果正確回答")
        print("\n💡 如果問題涉及私有文檔內容，有 RAG 的回答應該更準確、更具體！")
    elif no_rag_answer:
        print("\n⚠️  無 RAG 測試成功，但有 RAG 測試失敗")
        print("   請檢查檢索系統是否正常工作")
    elif rag_answer:
        print("\n⚠️  有 RAG 測試成功，但無 RAG 測試失敗")
        print("   請檢查 LLM 連接是否正常")
    else:
        print("\n❌ 兩個測試都失敗")
        print("   請檢查系統配置")


def main():
    """
    主程式：示範 hybrid search 的使用
    
    支援兩種分塊模式：
    - 字符分塊（預設）：快速、穩定
    - 語義分塊（可選）：更智能，能保持語義完整性
    
    可以通過環境變數 USE_SEMANTIC_CHUNKING=true 啟用語義分塊
    """
    
    print("=" * 60)
    print("Hybrid Search 系統初始化中...")
    print("使用 Hugging Face embedding 模型（完全免費，本地運行）")
    print("=" * 60)
    
    # [步驟 0] 可選：初始化共用的 Embedding 模型（用於語義分塊）
    # 檢查是否啟用語義分塊
    use_semantic_chunking = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"
    shared_embeddings = None
    
    if use_semantic_chunking:
        print("\n[0/6] 初始化共用的 Embedding 模型（用於語義分塊）...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from src.retrievers.vector_retriever import get_device
            
            # 設置 Hugging Face 模型緩存目錄（可選：外接硬碟路徑）
            hf_cache_dir = os.getenv("HF_CACHE_DIR", None)
            
            # 自動檢測設備（使用與 VectorRetriever 相同的邏輯）
            device = get_device()
            
            device_name_map = {
                'mps': 'MPS (macOS GPU)',
                'cuda': 'CUDA (NVIDIA GPU)',
                'cpu': 'CPU'
            }
            print(f"  使用設備: {device_name_map.get(device, device)}")
            
            # 構建 model_kwargs
            model_kwargs = {'device': device}
            if hf_cache_dir:
                model_kwargs['cache_dir'] = hf_cache_dir
                print(f"  模型緩存目錄: {hf_cache_dir}")
            
            # 初始化共用的 embedding 模型
            shared_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            print("  ✓ 共用 Embedding 模型初始化完成")
            print("  💡 此模型將同時用於語義分塊和向量檢索，節省內存和時間")
        except ImportError as e:
            print(f"  ⚠️  無法初始化語義分塊所需的依賴: {e}")
            print("  將回退到字符分塊模式")
            use_semantic_chunking = False
        except Exception as e:
            print(f"  ⚠️  初始化 Embedding 模型失敗: {e}")
            print("  將回退到字符分塊模式")
            use_semantic_chunking = False
    
    # 1. 初始化文檔處理器
    print("\n[1/6] 初始化文檔處理器...")
    if use_semantic_chunking and shared_embeddings:
        # 使用語義分塊模式
        processor = DocumentProcessor(
            embeddings=shared_embeddings,
            use_semantic_chunking=True,
            breakpoint_threshold_amount=1.5,  # 語義分塊敏感度
            min_chunk_size=100  # 最小 chunk 大小（字符數）
        )
    else:
        # 使用字符分塊模式（預設）
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # 2. 獲取 arXiv 論文
    print("\n[2/6] 從 arXiv 獲取論文...")
    print("  搜尋 AI、機器學習和自然語言處理相關論文...")
    papers = processor.fetch_papers(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",  # AI + 機器學習 + 自然語言處理
        max_results=40  # 增加到 40 篇以獲得更多樣化的結果
    )
    print(f"獲取了 {len(papers)} 篇論文")
    
    # 3. 處理文檔（分割成 chunks）
    print("\n[3/6] 處理文檔並分割成 chunks...")
    if use_semantic_chunking:
        print("  ⚠️  語義分塊需要計算 embedding，可能需要較長時間，請稍候...")
    documents = processor.process_documents(papers)
    print(f"總共創建了 {len(documents)} 個文檔 chunks")
    
    # 顯示一些範例文檔
    if documents:
        print("\n範例文檔（第一個 chunk）：")
        print(f"標題: {documents[0]['metadata']['title']}")
        chunking_method = documents[0]['metadata'].get('chunking_method', 'character')
        print(f"分塊方法: {chunking_method}")
        print(f"內容預覽: {documents[0]['content'][:200]}...")
    
    # 4. 初始化檢索器
    print("\n[4/6] 初始化檢索器...")
    
    # 稀疏檢索器 (BM25)
    print("  - 初始化稀疏檢索器 (BM25)...")
    bm25_retriever = BM25Retriever(documents)
    
    # 密集檢索器 (向量)
    print("  - 初始化密集檢索器（使用免費的 Hugging Face 模型）...")
    
    # 設置 Hugging Face 模型緩存目錄（可選：外接硬碟路徑）
    hf_cache_dir = os.getenv("HF_CACHE_DIR", None)
    
    # 如果使用語義分塊，傳入共用的 embeddings
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db",
        hf_cache_dir=hf_cache_dir,
        embeddings=shared_embeddings  # 傳入共用的 embeddings（如果有的話）
    )
    
    # 5. 初始化 Hybrid Search
    print("\n[5/6] 初始化 Hybrid Search...")
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
    
    # 9. 示範重排序功能（Reranker + RAGPipeline）
    print("\n" + "=" * 60)
    print("重排序功能示範 (Reranker + RAGPipeline)")
    print("=" * 60)
    
    try:
        # 初始化重排序器
        print("\n初始化重排序器...")
        reranker = Reranker(
            model_name="BAAI/bge-reranker-base",
            device=None,  # 自動檢測設備
            batch_size=16  # 批處理大小
        )
        
        # 初始化 RAG 管線
        print("初始化 RAG 管線...")
        rag_pipeline = RAGPipeline(
            hybrid_search=hybrid_search,
            reranker=reranker,
            recall_k=25,  # 第一階段召回 25 筆
            adaptive_recall=True,  # 啟用動態調整
            min_recall_k=10,
            max_recall_k=50
        )
        
        # 測試查詢
        test_query = "deep learning neural networks"
        print(f"\n測試查詢: '{test_query}'")
        print("-" * 60)
        
        # 使用 RAG 管線進行搜尋（包含重排序）
        results, stats = rag_pipeline.query(
            text=test_query,
            top_k=5,
            enable_rerank=True,
            return_stats=True
        )
        
        print(f"\n找到 {len(results)} 個結果（經過重排序）：\n")
        for i, result in enumerate(results, 1):
            print(f"結果 {i}:")
            print(f"  標題: {result['metadata']['title']}")
            print(f"  arXiv ID: {result['metadata']['arxiv_id']}")
            print(f"  重排序分數: {result.get('rerank_score', 'N/A'):.4f}")
            if 'hybrid_score' in result:
                print(f"  原始混合分數: {result['hybrid_score']:.4f}")
            print(f"  內容預覽: {result['content'][:150]}...")
            print()
        
        # 顯示性能統計
        print("\n性能統計:")
        print(f"  召回階段耗時: {stats['recall_time']:.2f}s")
        print(f"  重排階段耗時: {stats['rerank_time']:.2f}s")
        print(f"  總耗時: {stats['total_time']:.2f}s")
        print(f"  實際召回數量: {stats['recall_k']} 筆")
        print(f"  候選結果數: {stats['candidates_found']} 筆")
        print(f"  最終結果數: {stats['final_results']} 筆")
        
        # 比較：有重排序 vs 無重排序
        print("\n" + "-" * 60)
        print("比較：有重排序 vs 無重排序")
        print("-" * 60)
        
        # 無重排序
        print("\n[無重排序] 僅使用 Hybrid Search:")
        no_rerank_results = hybrid_search.retrieve(test_query, top_k=5)
        for i, result in enumerate(no_rerank_results, 1):
            print(f"{i}. {result['metadata']['title']} (混合分數: {result.get('hybrid_score', 0.0):.4f})")
        
        # 有重排序
        print("\n[有重排序] 使用 RAG Pipeline:")
        rerank_results = rag_pipeline.query(test_query, top_k=5, enable_rerank=True)
        for i, result in enumerate(rerank_results, 1):
            rerank_score = result.get('rerank_score', 0.0)
            hybrid_score = result.get('hybrid_score', 0.0)
            print(f"{i}. {result['metadata']['title']}")
            print(f"   重排序分數: {rerank_score:.4f}, 原始混合分數: {hybrid_score:.4f}")
        
        # 顯示累積統計
        print("\n" + "-" * 60)
        print("累積性能統計:")
        print("-" * 60)
        cumulative_stats = rag_pipeline.get_stats()
        print(f"  總查詢數: {cumulative_stats['total_queries']}")
        print(f"  平均召回時間: {cumulative_stats['avg_recall_time']:.2f}s")
        print(f"  平均重排時間: {cumulative_stats['avg_rerank_time']:.2f}s")
        print(f"  平均總時間: {cumulative_stats['avg_total_time']:.2f}s")
        
        # 10. 執行全面的 RAG 功效測試
        print("\n" + "=" * 60)
        print("執行全面的 RAG 功效測試")
        print("=" * 60)
        
        comprehensive_rag_test(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            hybrid_search=hybrid_search,
            rag_pipeline=rag_pipeline
        )
        
        # 11. 示範 Prompt 格式化和 LLM 集成
        print("\n" + "=" * 60)
        print("Prompt 格式化和 LLM 集成示範")
        print("=" * 60)
        
        try:
            # 初始化 Prompt 格式化器
            print("\n初始化 Prompt 格式化器...")
            formatter = PromptFormatter(
                include_metadata=True,
                format_style="detailed"
            )
            
            # 測試查詢
            test_query_llm = "How do transformer models work?"
            print(f"\n測試查詢: '{test_query_llm}'")
            print("-" * 60)
            
            # 檢索相關文檔
            llm_results = rag_pipeline.query(
                text=test_query_llm,
                top_k=3,
                enable_rerank=True
            )
            
            if llm_results:
                # 格式化檢索結果（arXiv 論文使用 "paper" 類型）
                print("\n格式化檢索結果...")
                formatted_context = formatter.format_context(llm_results, document_type="paper")
                print("\n格式化後的上下文（前 500 字符）：")
                print(formatted_context[:500] + "...")
                
                # 創建完整的 prompt（arXiv 論文使用 "paper" 類型）
                full_prompt = formatter.create_prompt(test_query_llm, formatted_context, document_type="paper")
                print("\n完整 Prompt 長度:", len(full_prompt), "字符")
                
                # 嘗試使用 Ollama LLM
                print("\n" + "-" * 60)
                print("嘗試連接 Ollama LLM...")
                print("-" * 60)
                
                try:
                    # 顯示推薦的模型
                    OllamaLLM.print_recommended_models()
                    
                    # 初始化 LLM（使用推薦的小模型）
                    llm = OllamaLLM(
                        model_name="llama3.2:3b",  # 適合 16GB 內存
                        timeout=180
                    )
                    
                    print(f"\n使用模型: {llm.model_name}")
                    print("生成回答中（這可能需要一些時間）...\n")
                    print("-" * 60)
                    
                    # 生成回答
                    answer = llm.generate(
                        prompt=full_prompt,
                        temperature=0.7,
                        max_tokens=2048,
                        stream=False
                    )
                    
                    print("\n" + "-" * 60)
                    print("LLM 生成的回答：")
                    print("-" * 60)
                    print(answer)
                    
                except ConnectionError as e:
                    print(f"\n⚠️  Ollama 連接錯誤: {e}")
                    print("\n請按照以下步驟設置 Ollama：")
                    print("  1. 安裝 Ollama: https://ollama.ai/download")
                    print("  2. 啟動 Ollama 服務（通常會自動啟動）")
                    print("  3. 下載模型: ollama pull llama3.2:3b")
                    print("  4. 重新運行此程序")
                except Exception as e:
                    print(f"\n⚠️  LLM 生成出錯: {e}")
                    print("您可以繼續使用格式化功能，只是不生成 LLM 回答。")
                
                # 顯示格式化後的 prompt（即使 LLM 失敗）
                print("\n" + "-" * 60)
                print("格式化後的 Prompt（可用於手動測試）：")
                print("-" * 60)
                print(full_prompt[:1000] + f"...\n（已截斷，完整長度: {len(full_prompt)} 字符）")
            else:
                print("⚠️  未找到相關文檔，無法進行格式化")
                
        except Exception as e:
            print(f"\n⚠️  Prompt 格式化出錯: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"\n⚠️  重排序功能需要額外依賴: {e}")
        print("請確保已安裝 sentence-transformers: pip install sentence-transformers")
    except Exception as e:
        print(f"\n⚠️  重排序功能出錯: {e}")
        print("這可能是因為模型下載失敗或設備不支持。")
        print("您可以繼續使用 Hybrid Search 功能。")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    
    # 提示：如何啟用語義分塊
    if not use_semantic_chunking:
        print("\n💡 提示：要啟用語義分塊模式，請設置環境變數：")
        print("  export USE_SEMANTIC_CHUNKING=true")
        print("  或")
        print("  USE_SEMANTIC_CHUNKING=true python main.py")
        print("\n語義分塊的優點：")
        print("  - 能保持語義完整性，不會在句子中間切分")
        print("  - 根據語義相似度自動決定切分點")
        print("  - 可能提升檢索效果")
        print("\n注意：語義分塊需要安裝 langchain-experimental：")
        print("  pip install langchain-experimental")


if __name__ == "__main__":
    main()

