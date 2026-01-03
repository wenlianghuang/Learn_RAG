"""
Sub-query Decomposition RAG 測試腳本
示範如何使用子問題拆解來提升 RAG 系統的效果
"""
import os
import sys
import time
import hashlib
from src import (
    DocumentProcessor,
    BM25Retriever,
    VectorRetriever,
    HybridSearch,
    Reranker,
    RAGPipeline,
    PromptFormatter,
    OllamaLLM,
    SubQueryDecompositionRAG
)


def test_subquery_rag_with_papers():
    """使用 arXiv 論文測試 Sub-query Decomposition RAG"""
    print("=" * 60)
    print("Sub-query Decomposition RAG 測試")
    print("=" * 60)
    
    # 1. 初始化文檔處理器
    print("\n[1/7] 初始化文檔處理器...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # 2. 獲取論文
    print("\n[2/7] 從 arXiv 獲取論文...")
    papers = processor.fetch_papers(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        max_results=20
    )
    print(f"✅ 獲取了 {len(papers)} 篇論文")
    
    # 3. 處理文檔
    print("\n[3/7] 處理文檔並分割成 chunks...")
    documents = processor.process_documents(papers)
    print(f"✅ 總共創建了 {len(documents)} 個文檔 chunks")
    
    # 4. 初始化檢索器
    print("\n[4/7] 初始化檢索器...")
    bm25_retriever = BM25Retriever(documents)
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db_subquery"
    )
    hybrid_search = HybridSearch(
        sparse_retriever=bm25_retriever,
        dense_retriever=vector_retriever,
        fusion_method="rrf",
        rrf_k=60
    )
    
    # 5. 初始化 RAG 管線
    print("\n[5/7] 初始化 RAG 管線...")
    reranker = Reranker(
        model_name="BAAI/bge-reranker-base",
        device=None,
        batch_size=16
    )
    rag_pipeline = RAGPipeline(
        hybrid_search=hybrid_search,
        reranker=reranker,
        recall_k=25,
        adaptive_recall=True
    )
    
    # 6. 初始化 LLM 和格式化器
    print("\n[6/7] 初始化 LLM 和格式化器...")
    try:
        llm = OllamaLLM(model_name="llama3.2:3b", timeout=180)
        print(f"✅ LLM 初始化完成: {llm.model_name}")
    except Exception as e:
        print(f"⚠️  LLM 初始化失敗: {e}")
        print("請確保 Ollama 正在運行並已下載模型")
        return
    
    formatter = PromptFormatter(
        include_metadata=True,
        format_style="detailed"
    )
    
    # 7. 初始化 Sub-query Decomposition RAG
    print("\n[7/7] 初始化 Sub-query Decomposition RAG...")
    subquery_rag = SubQueryDecompositionRAG(
        rag_pipeline=rag_pipeline,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        enable_parallel=True
    )
    print("✅ 系統初始化完成！")
    
    # 測試查詢
    print("\n" + "=" * 60)
    print("開始測試")
    print("=" * 60)
    
    test_queries = [
        "transformer architecture and attention mechanism",
        "比較深度學習和機器學習的差異",
        "How do neural networks learn and optimize?",
    ]
    
    for query in test_queries:
        print("\n" + "-" * 60)
        print(f"測試查詢: '{query}'")
        print("-" * 60)
        
        try:
            # 使用 Sub-query Decomposition RAG
            result = subquery_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_sub_queries=True
            )
            
            # 顯示結果
            print(f"\n📊 查詢統計:")
            print(f"   總耗時: {result['total_time']:.2f}s")
            print(f"   檢索耗時: {result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {result.get('answer_time', 0):.2f}s")
            print(f"   找到文檔數: {result['total_docs_found']}")
            
            if result.get('sub_queries'):
                print(f"\n🔍 生成的子問題 ({len(result['sub_queries'])} 個):")
                for i, sq in enumerate(result['sub_queries'], 1):
                    print(f"   {i}. {sq}")
            
            print(f"\n📚 檢索到的文檔 (前 3 個):")
            for i, doc in enumerate(result['results'][:3], 1):
                print(f"\n   {i}. {doc['metadata'].get('title', 'N/A')}")
                print(f"      arXiv ID: {doc['metadata'].get('arxiv_id', 'N/A')}")
                score = doc.get('rerank_score', doc.get('hybrid_score', 0))
                print(f"      相關性分數: {score:.4f}")
                print(f"      內容預覽: {doc['content'][:100]}...")
            
            print(f"\n🤖 生成的回答:")
            print("-" * 40)
            print(result['answer'])
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ 查詢處理出錯: {e}")
            import traceback
            traceback.print_exc()
            continue


def test_subquery_vs_normal_rag():
    """對比測試：Sub-query Decomposition vs 正常 RAG"""
    print("\n" + "=" * 60)
    print("對比測試：Sub-query Decomposition vs 正常 RAG")
    print("=" * 60)
    
    # 1. 初始化系統（與 test_subquery_rag_with_papers 相同）
    print("\n[初始化系統]")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    print("從 arXiv 獲取論文...")
    papers = processor.fetch_papers(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        max_results=20
    )
    print(f"✅ 獲取了 {len(papers)} 篇論文")
    
    documents = processor.process_documents(papers)
    print(f"✅ 總共創建了 {len(documents)} 個文檔 chunks")
    
    # 初始化檢索器
    print("初始化檢索器...")
    bm25_retriever = BM25Retriever(documents)
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db_compare"
    )
    hybrid_search = HybridSearch(
        sparse_retriever=bm25_retriever,
        dense_retriever=vector_retriever,
        fusion_method="rrf",
        rrf_k=60
    )
    
    reranker = Reranker(
        model_name="BAAI/bge-reranker-base",
        device=None,
        batch_size=16
    )
    rag_pipeline = RAGPipeline(
        hybrid_search=hybrid_search,
        reranker=reranker,
        recall_k=25,
        adaptive_recall=True
    )
    
    # 初始化 LLM 和格式化器
    print("初始化 LLM 和格式化器...")
    try:
        llm = OllamaLLM(model_name="llama3.2:3b", timeout=180)
        print(f"✅ LLM 初始化完成: {llm.model_name}")
    except Exception as e:
        print(f"⚠️  LLM 初始化失敗: {e}")
        print("請確保 Ollama 正在運行並已下載模型")
        return
    
    formatter = PromptFormatter(
        include_metadata=True,
        format_style="detailed"
    )
    
    # 初始化 Sub-query Decomposition RAG
    print("初始化 Sub-query Decomposition RAG...")
    subquery_rag = SubQueryDecompositionRAG(
        rag_pipeline=rag_pipeline,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        enable_parallel=True
    )
    print("✅ 系統初始化完成！")
    
    # 2. 測試查詢（選擇複雜的查詢以突出差異）
    test_queries = [
        "transformer architecture, attention mechanism, and optimization techniques",
        "比較深度學習和機器學習的差異、優缺點和應用場景",
        "How do neural networks learn, optimize, and generalize?",
    ]
    
    print("\n" + "=" * 60)
    print("開始對比測試")
    print("=" * 60)
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"測試查詢: '{query}'")
        print("=" * 60)
        
        # === 方法 1: 正常 RAG ===
        print("\n[方法 1] 正常 RAG")
        print("-" * 60)
        
        normal_start = time.time()
        try:
            # 正常 RAG：直接使用 RAGPipeline
            normal_results = rag_pipeline.query(
                text=query,
                top_k=5,
                enable_rerank=True
            )
            
            # 格式化並生成答案
            normal_context = formatter.format_context(
                normal_results,
                document_type="paper"
            )
            normal_prompt = formatter.create_prompt(
                query,
                normal_context,
                document_type="paper"
            )
            
            normal_answer_start = time.time()
            normal_answer = llm.generate(
                prompt=normal_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            normal_answer_time = time.time() - normal_answer_start
            normal_total_time = time.time() - normal_start
            
            print(f"✅ 正常 RAG 完成")
            print(f"   檢索耗時: {normal_total_time - normal_answer_time:.2f}s")
            print(f"   生成耗時: {normal_answer_time:.2f}s")
            print(f"   總耗時: {normal_total_time:.2f}s")
            print(f"   找到文檔數: {len(normal_results)}")
            
            # 顯示檢索結果
            print(f"\n   檢索到的文檔 (前 3 個):")
            for i, doc in enumerate(normal_results[:3], 1):
                score = doc.get('rerank_score', doc.get('hybrid_score', 0))
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
        except Exception as e:
            print(f"❌ 正常 RAG 出錯: {e}")
            normal_answer = None
            normal_total_time = 0
            normal_results = []
            import traceback
            traceback.print_exc()
        
        # === 方法 2: Sub-query Decomposition RAG ===
        print("\n[方法 2] Sub-query Decomposition RAG")
        print("-" * 60)
        
        try:
            subquery_result = subquery_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_sub_queries=True
            )
            
            print(f"✅ Sub-query RAG 完成")
            print(f"   檢索耗時: {subquery_result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {subquery_result.get('answer_time', 0):.2f}s")
            print(f"   總耗時: {subquery_result['total_time']:.2f}s")
            print(f"   找到文檔數: {subquery_result['total_docs_found']}")
            
            if subquery_result.get('sub_queries'):
                print(f"\n   生成的子問題 ({len(subquery_result['sub_queries'])} 個):")
                for i, sq in enumerate(subquery_result['sub_queries'], 1):
                    print(f"   {i}. {sq}")
            
            # 顯示檢索結果
            print(f"\n   檢索到的文檔 (前 3 個):")
            for i, doc in enumerate(subquery_result['results'][:3], 1):
                score = doc.get('rerank_score', doc.get('hybrid_score', 0))
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
        except Exception as e:
            print(f"❌ Sub-query RAG 出錯: {e}")
            subquery_result = None
            import traceback
            traceback.print_exc()
        
        # === 對比總結 ===
        print("\n" + "-" * 60)
        print("[對比總結]")
        print("-" * 60)
        
        if normal_answer and subquery_result:
            print(f"\n📊 性能對比:")
            print(f"   正常 RAG 總耗時: {normal_total_time:.2f}s")
            print(f"   Sub-query RAG 總耗時: {subquery_result['total_time']:.2f}s")
            time_diff = abs(normal_total_time - subquery_result['total_time'])
            time_ratio = (subquery_result['total_time'] / normal_total_time * 100) if normal_total_time > 0 else 0
            print(f"   時間差異: {time_diff:.2f}s (Sub-query 是正常 RAG 的 {time_ratio:.1f}%)")
            
            print(f"\n📚 檢索結果對比:")
            print(f"   正常 RAG 文檔數: {len(normal_results)}")
            print(f"   Sub-query RAG 文檔數: {subquery_result['total_docs_found']}")
            
            # 檢查文檔重疊度
            normal_doc_ids = set()
            for doc in normal_results:
                metadata = doc.get('metadata', {})
                if 'arxiv_id' in metadata and 'chunk_index' in metadata:
                    normal_doc_ids.add(f"{metadata['arxiv_id']}_{metadata['chunk_index']}")
                else:
                    # 回退到內容 hash
                    content = doc.get('content', '')
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                    normal_doc_ids.add(f"doc_{content_hash}")
            
            subquery_doc_ids = set()
            for doc in subquery_result['results']:
                metadata = doc.get('metadata', {})
                if 'arxiv_id' in metadata and 'chunk_index' in metadata:
                    subquery_doc_ids.add(f"{metadata['arxiv_id']}_{metadata['chunk_index']}")
                else:
                    # 回退到內容 hash
                    content = doc.get('content', '')
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                    subquery_doc_ids.add(f"doc_{content_hash}")
            
            overlap = len(normal_doc_ids & subquery_doc_ids)
            total_unique = len(normal_doc_ids | subquery_doc_ids)
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            print(f"   文檔重疊數: {overlap}/{total_unique}")
            print(f"   文檔重疊率: {overlap_ratio:.2%}")
            
            print(f"\n💡 觀察:")
            if subquery_result['total_docs_found'] > len(normal_results):
                print("   ✓ Sub-query RAG 找到了更多文檔（可能覆蓋更全面）")
            elif subquery_result['total_docs_found'] < len(normal_results):
                print("   - 正常 RAG 找到了更多文檔")
            else:
                print("   = 兩種方法找到的文檔數量相同")
            
            if overlap_ratio < 0.5:
                print("   ✓ 兩種方法找到的文檔差異較大（Sub-query 可能從不同角度檢索）")
            elif overlap_ratio > 0.8:
                print("   = 兩種方法找到的文檔高度重疊")
            else:
                print("   ~ 兩種方法找到的文檔有部分重疊")
            
            if subquery_result['total_time'] > normal_total_time * 1.5:
                print(f"   ⚠ Sub-query RAG 耗時較長（因為需要生成 {len(subquery_result.get('sub_queries', []))} 個子問題）")
            elif subquery_result['total_time'] < normal_total_time:
                print("   ✓ Sub-query RAG 耗時更短（可能因為並行處理）")
            else:
                print("   = 兩種方法耗時相近")
            
            # 顯示答案長度對比（簡單的質量指標）
            normal_answer_len = len(normal_answer) if normal_answer else 0
            subquery_answer_len = len(subquery_result.get('answer', ''))
            print(f"\n📝 答案長度對比:")
            print(f"   正常 RAG 答案長度: {normal_answer_len} 字符")
            print(f"   Sub-query RAG 答案長度: {subquery_answer_len} 字符")
        
        print("\n" + "=" * 60)


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="測試 Sub-query Decomposition RAG")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="執行對比測試（Sub-query vs 正常 RAG）"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        test_subquery_vs_normal_rag()
    else:
        test_subquery_rag_with_papers()


if __name__ == "__main__":
    main()

