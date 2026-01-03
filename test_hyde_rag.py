"""
HyDE RAG 測試腳本：測試 HyDE 並與傳統 RAG 進行對比
"""
import os
import sys
import time
from src import (
    DocumentProcessor,
    BM25Retriever,
    VectorRetriever,
    HybridSearch,
    Reranker,
    RAGPipeline,
    PromptFormatter,
    OllamaLLM,
    HyDERAG,
    SubQueryDecompositionRAG,
    HybridSubqueryHyDERAG
)


def test_hyde_vs_normal_rag():
    """對比測試：HyDE RAG vs 正常 RAG"""
    print("=" * 60)
    print("HyDE RAG vs 正常 RAG 對比測試")
    print("=" * 60)
    
    # 1. 初始化系統
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
        persist_directory="./chroma_db_hyde"
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
    
    # 初始化 HyDE RAG
    print("初始化 HyDE RAG...")
    hyde_rag = HyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        hypothetical_length=200,
        temperature=0.7
    )
    print("✅ 系統初始化完成！")
    
    # 2. 測試查詢
    test_queries = [
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
        "How do neural networks learn and optimize?",
        "深度學習中的反向傳播算法原理",
    ]
    
    print("\n" + "=" * 60)
    print("開始對比測試")
    print("=" * 60)
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"測試查詢: '{query}'")
        print("=" * 60)
        
        # === 方法 1: 正常 RAG（使用原始問題檢索）===
        print("\n[方法 1] 正常 RAG（原始問題檢索）")
        print("-" * 60)
        
        normal_start = time.time()
        try:
            # 正常 RAG：直接使用原始問題檢索
            normal_results = vector_retriever.retrieve(
                query=query,  # 使用原始問題
                top_k=5
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
                score = doc.get('score', 0)
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
        
        # === 方法 2: HyDE RAG（使用假設性文檔檢索）===
        print("\n[方法 2] HyDE RAG（假設性文檔檢索）")
        print("-" * 60)
        
        try:
            hyde_result = hyde_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_hypothetical=True
            )
            
            print(f"✅ HyDE RAG 完成")
            print(f"   假設性文檔生成: {hyde_result.get('hypothetical_time', 0):.2f}s")
            print(f"   檢索耗時: {hyde_result.get('retrieval_time', 0):.2f}s")
            print(f"   生成耗時: {hyde_result.get('answer_time', 0):.2f}s")
            print(f"   總耗時: {hyde_result['total_time']:.2f}s")
            print(f"   找到文檔數: {hyde_result['total_docs_found']}")
            
            if hyde_result.get('hypothetical_document'):
                print(f"\n   生成的假設性文檔:")
                hypo_doc = hyde_result['hypothetical_document']
                print(f"   {hypo_doc[:200]}...")
                print(f"   (完整長度: {len(hypo_doc)} 字符)")
            
            # 顯示檢索結果
            print(f"\n   檢索到的文檔 (前 3 個):")
            for i, doc in enumerate(hyde_result['results'][:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
        except Exception as e:
            print(f"❌ HyDE RAG 出錯: {e}")
            hyde_result = None
            import traceback
            traceback.print_exc()
        
        # === 對比總結 ===
        print("\n" + "-" * 60)
        print("[對比總結]")
        print("-" * 60)
        
        if normal_answer and hyde_result:
            print(f"\n📊 性能對比:")
            print(f"   正常 RAG 總耗時: {normal_total_time:.2f}s")
            print(f"   HyDE RAG 總耗時: {hyde_result['total_time']:.2f}s")
            time_diff = abs(normal_total_time - hyde_result['total_time'])
            time_ratio = (hyde_result['total_time'] / normal_total_time * 100) if normal_total_time > 0 else 0
            print(f"   時間差異: {time_diff:.2f}s (HyDE 是正常 RAG 的 {time_ratio:.1f}%)")
            
            print(f"\n📚 檢索結果對比:")
            print(f"   正常 RAG 文檔數: {len(normal_results)}")
            print(f"   HyDE RAG 文檔數: {hyde_result['total_docs_found']}")
            
            # 檢查文檔重疊度
            normal_doc_ids = set()
            for doc in normal_results:
                metadata = doc.get('metadata', {})
                if 'arxiv_id' in metadata and 'chunk_index' in metadata:
                    normal_doc_ids.add(f"{metadata['arxiv_id']}_{metadata['chunk_index']}")
                else:
                    import hashlib
                    content = doc.get('content', '')
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                    normal_doc_ids.add(f"doc_{content_hash}")
            
            hyde_doc_ids = set()
            for doc in hyde_result['results']:
                metadata = doc.get('metadata', {})
                if 'arxiv_id' in metadata and 'chunk_index' in metadata:
                    hyde_doc_ids.add(f"{metadata['arxiv_id']}_{metadata['chunk_index']}")
                else:
                    import hashlib
                    content = doc.get('content', '')
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                    hyde_doc_ids.add(f"doc_{content_hash}")
            
            overlap = len(normal_doc_ids & hyde_doc_ids)
            total_unique = len(normal_doc_ids | hyde_doc_ids)
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            print(f"   文檔重疊數: {overlap}/{total_unique}")
            print(f"   文檔重疊率: {overlap_ratio:.2%}")
            
            # 比較分數
            if normal_results and hyde_result['results']:
                normal_avg_score = sum(doc.get('score', 0) for doc in normal_results) / len(normal_results)
                hyde_avg_score = sum(doc.get('score', 0) for doc in hyde_result['results']) / len(hyde_result['results'])
                print(f"\n   平均相關性分數:")
                print(f"   正常 RAG: {normal_avg_score:.4f}")
                print(f"   HyDE RAG: {hyde_avg_score:.4f}")
                if hyde_avg_score > normal_avg_score:
                    improvement = ((hyde_avg_score - normal_avg_score) / normal_avg_score * 100) if normal_avg_score > 0 else 0
                    print(f"   HyDE 提升: +{improvement:.1f}%")
            
            print(f"\n💡 觀察:")
            if hyde_result['total_docs_found'] > len(normal_results):
                print("   ✓ HyDE RAG 找到了更多文檔")
            elif hyde_result['total_docs_found'] < len(normal_results):
                print("   - 正常 RAG 找到了更多文檔")
            else:
                print("   = 兩種方法找到的文檔數量相同")
            
            if overlap_ratio < 0.5:
                print("   ✓ 兩種方法找到的文檔差異較大（HyDE 可能找到了不同的相關文檔）")
            elif overlap_ratio > 0.8:
                print("   = 兩種方法找到的文檔高度重疊")
            else:
                print("   ~ 兩種方法找到的文檔有部分重疊")
            
            if hyde_result['total_time'] > normal_total_time * 1.2:
                print(f"   ⚠ HyDE RAG 耗時較長（因為需要生成假設性文檔）")
            elif hyde_result['total_time'] < normal_total_time:
                print("   ✓ HyDE RAG 耗時更短")
            else:
                print("   = 兩種方法耗時相近")
            
            # 顯示答案長度對比
            normal_answer_len = len(normal_answer) if normal_answer else 0
            hyde_answer_len = len(hyde_result.get('answer', ''))
            print(f"\n📝 答案長度對比:")
            print(f"   正常 RAG 答案長度: {normal_answer_len} 字符")
            print(f"   HyDE RAG 答案長度: {hyde_answer_len} 字符")
        
        print("\n" + "=" * 60)


def test_hyde_basic():
    """基本 HyDE 功能測試"""
    print("=" * 60)
    print("HyDE RAG 基本功能測試")
    print("=" * 60)
    
    # 初始化系統（與對比測試相同）
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    papers = processor.fetch_papers(
        query="cat:cs.AI OR cat:cs.LG",
        max_results=10
    )
    documents = processor.process_documents(papers)
    
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db_hyde_basic"
    )
    
    hybrid_search = HybridSearch(
        sparse_retriever=BM25Retriever(documents),
        dense_retriever=vector_retriever,
        fusion_method="rrf"
    )
    
    rag_pipeline = RAGPipeline(
        hybrid_search=hybrid_search,
        reranker=Reranker(),
        recall_k=25
    )
    
    llm = OllamaLLM(model_name="llama3.2:3b", timeout=180)
    formatter = PromptFormatter()
    
    hyde_rag = HyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm
    )
    
    # 測試查詢
    question = "什麼是區塊鏈的共識機制？"
    print(f"\n測試查詢: '{question}'")
    print("-" * 60)
    
    result = hyde_rag.generate_answer(
        question=question,
        formatter=formatter,
        top_k=5,
        return_hypothetical=True
    )
    
    print(f"\n📊 結果:")
    print(f"   總耗時: {result['total_time']:.2f}s")
    print(f"   找到文檔數: {result['total_docs_found']}")
    
    if result.get('hypothetical_document'):
        print(f"\n🔍 生成的假設性文檔:")
        print("-" * 40)
        print(result['hypothetical_document'])
        print("-" * 40)
    
    print(f"\n📚 檢索到的文檔:")
    for i, doc in enumerate(result['results'][:3], 1):
        print(f"\n   {i}. {doc['metadata'].get('title', 'N/A')}")
        print(f"      分數: {doc.get('score', 0):.4f}")
        print(f"      內容預覽: {doc['content'][:150]}...")
    
    print(f"\n🤖 生成的回答:")
    print("-" * 40)
    print(result['answer'])
    print("-" * 40)


def test_hybrid_vs_all_methods():
    """對比測試：融合方法 vs 所有單獨方法"""
    print("=" * 60)
    print("融合方法對比測試：Hybrid (Sub-query + HyDE) vs 所有方法")
    print("=" * 60)
    
    # 1. 初始化系統
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
        persist_directory="./chroma_db_hybrid"
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
    
    # 初始化所有方法
    print("初始化所有 RAG 方法...")
    subquery_rag = SubQueryDecompositionRAG(
        rag_pipeline=rag_pipeline,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        enable_parallel=True
    )
    
    hyde_rag = HyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        hypothetical_length=200,
        temperature=0.7
    )
    
    hybrid_rag = HybridSubqueryHyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        hypothetical_length=200,
        temperature_subquery=0.3,
        temperature_hyde=0.7,
        enable_parallel=True
    )
    print("✅ 系統初始化完成！")
    
    # 2. 測試查詢
    test_queries = [
        "transformer architecture, attention mechanism, and optimization techniques",
        "比較深度學習和機器學習的差異、優缺點和應用場景",
    ]
    
    print("\n" + "=" * 60)
    print("開始對比測試")
    print("=" * 60)
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"測試查詢: '{query}'")
        print("=" * 60)
        
        results = {}
        
        # === 方法 1: 正常 RAG ===
        print("\n[方法 1] 正常 RAG")
        print("-" * 60)
        try:
            normal_start = time.time()
            normal_results = vector_retriever.retrieve(query=query, top_k=5)
            normal_time = time.time() - normal_start
            results['normal'] = {
                'docs': normal_results,
                'time': normal_time,
                'count': len(normal_results)
            }
            print(f"✅ 找到 {len(normal_results)} 個文檔，耗時 {normal_time:.2f}s")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['normal'] = {'docs': [], 'time': 0, 'count': 0}
        
        # === 方法 2: Sub-query RAG ===
        print("\n[方法 2] Sub-query Decomposition RAG")
        print("-" * 60)
        try:
            subquery_result = subquery_rag.query(
                question=query,
                top_k=5,
                return_sub_queries=True
            )
            results['subquery'] = {
                'docs': subquery_result['results'],
                'time': subquery_result['elapsed_time'],
                'count': subquery_result['total_docs_found'],
                'sub_queries': subquery_result.get('sub_queries', [])
            }
            print(f"✅ 找到 {subquery_result['total_docs_found']} 個文檔，耗時 {subquery_result['elapsed_time']:.2f}s")
            if subquery_result.get('sub_queries'):
                print(f"   子問題數: {len(subquery_result['sub_queries'])}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['subquery'] = {'docs': [], 'time': 0, 'count': 0, 'sub_queries': []}
        
        # === 方法 3: HyDE RAG ===
        print("\n[方法 3] HyDE RAG")
        print("-" * 60)
        try:
            hyde_result = hyde_rag.query(
                question=query,
                top_k=5,
                return_hypothetical=True
            )
            results['hyde'] = {
                'docs': hyde_result['results'],
                'time': hyde_result['elapsed_time'],
                'count': hyde_result['total_docs_found'],
                'hypothetical': hyde_result.get('hypothetical_document', '')
            }
            print(f"✅ 找到 {hyde_result['total_docs_found']} 個文檔，耗時 {hyde_result['elapsed_time']:.2f}s")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['hyde'] = {'docs': [], 'time': 0, 'count': 0, 'hypothetical': ''}
        
        # === 方法 4: Hybrid (Sub-query + HyDE) RAG ===
        print("\n[方法 4] Hybrid (Sub-query + HyDE) RAG")
        print("-" * 60)
        try:
            hybrid_result = hybrid_rag.query(
                question=query,
                top_k=5,
                return_sub_queries=True,
                return_hypothetical=True
            )
            results['hybrid'] = {
                'docs': hybrid_result['results'],
                'time': hybrid_result['elapsed_time'],
                'count': hybrid_result['total_docs_found'],
                'sub_queries': hybrid_result.get('sub_queries', []),
                'hypothetical': hybrid_result.get('hypothetical_documents', {})
            }
            print(f"✅ 找到 {hybrid_result['total_docs_found']} 個文檔，耗時 {hybrid_result['elapsed_time']:.2f}s")
            if hybrid_result.get('sub_queries'):
                print(f"   子問題數: {len(hybrid_result['sub_queries'])}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            import traceback
            traceback.print_exc()
            results['hybrid'] = {'docs': [], 'time': 0, 'count': 0, 'sub_queries': [], 'hypothetical': {}}
        
        # === 對比總結 ===
        print("\n" + "-" * 60)
        print("[對比總結]")
        print("-" * 60)
        
        print(f"\n📊 性能對比:")
        for method_name, method_result in results.items():
            method_display = {
                'normal': '正常 RAG',
                'subquery': 'Sub-query RAG',
                'hyde': 'HyDE RAG',
                'hybrid': 'Hybrid RAG'
            }.get(method_name, method_name)
            print(f"   {method_display}: {method_result['time']:.2f}s, {method_result['count']} 個文檔")
        
        # 比較平均分數
        print(f"\n📈 平均相關性分數:")
        for method_name, method_result in results.items():
            if method_result['docs']:
                avg_score = sum(doc.get('score', 0) for doc in method_result['docs']) / len(method_result['docs'])
                method_display = {
                    'normal': '正常 RAG',
                    'subquery': 'Sub-query RAG',
                    'hyde': 'HyDE RAG',
                    'hybrid': 'Hybrid RAG'
                }.get(method_name, method_name)
                print(f"   {method_display}: {avg_score:.4f}")
        
        # 文檔重疊分析
        print(f"\n📚 文檔重疊分析:")
        if results['hybrid']['docs']:
            # 獲取 Hybrid 方法的文檔 ID
            hybrid_doc_ids = set()
            for doc in results['hybrid']['docs']:
                metadata = doc.get('metadata', {})
                if 'arxiv_id' in metadata and 'chunk_index' in metadata:
                    hybrid_doc_ids.add(f"{metadata['arxiv_id']}_{metadata['chunk_index']}")
            
            # 與其他方法比較
            for method_name in ['normal', 'subquery', 'hyde']:
                if results[method_name]['docs']:
                    method_doc_ids = set()
                    for doc in results[method_name]['docs']:
                        metadata = doc.get('metadata', {})
                        if 'arxiv_id' in metadata and 'chunk_index' in metadata:
                            method_doc_ids.add(f"{metadata['arxiv_id']}_{metadata['chunk_index']}")
                    
                    overlap = len(hybrid_doc_ids & method_doc_ids)
                    method_display = {
                        'normal': '正常 RAG',
                        'subquery': 'Sub-query RAG',
                        'hyde': 'HyDE RAG'
                    }.get(method_name, method_name)
                    print(f"   Hybrid vs {method_display}: {overlap} 個重疊")
        
        print(f"\n💡 觀察:")
        hybrid_count = results['hybrid']['count']
        hybrid_time = results['hybrid']['time']
        
        if hybrid_count >= max(r['count'] for r in results.values() if r['count'] > 0):
            print("   ✓ Hybrid 方法找到了最多或最多的文檔")
        
        if hybrid_time > max(r['time'] for r in results.values() if r['time'] > 0) * 0.8:
            print("   ⚠ Hybrid 方法耗時較長（因為結合了兩種方法）")
        else:
            print("   ✓ Hybrid 方法性能可接受")
        
        print("\n" + "=" * 60)


def evaluate_answer_quality(answer: str, query: str) -> dict:
    """
    評估答案質量
    
    Args:
        answer: 生成的答案
        query: 原始問題
        
    Returns:
        包含各項評分的字典
    """
    query_keywords = set(query.lower().split())
    answer_lower = answer.lower()
    
    # 關鍵詞覆蓋率
    matched = sum(1 for kw in query_keywords if kw in answer_lower)
    keyword_coverage = matched / len(query_keywords) if query_keywords else 0
    
    # 答案詳細程度（長度）
    detail_score = min(len(answer) / 500, 1.0)  # 500 字符為滿分
    
    # 專業術語數量（簡單啟發式）
    technical_terms = ['algorithm', 'mechanism', 'architecture', 'optimization', 
                      'technique', 'method', 'model', 'system', 'process',
                      '算法', '機制', '架構', '優化', '技術', '方法', '模型', '系統']
    tech_count = sum(1 for term in technical_terms if term in answer_lower)
    tech_score = min(tech_count / 5, 1.0)
    
    overall_score = (keyword_coverage * 0.4 + detail_score * 0.3 + tech_score * 0.3)
    
    return {
        'keyword_coverage': keyword_coverage,
        'detail_score': detail_score,
        'tech_score': tech_score,
        'overall_score': overall_score
    }


def test_visual_comparison_with_answers():
    """視覺化對比測試：包含實際答案，讓用戶直觀感受差異"""
    print("=" * 80)
    print("🎯 視覺化 RAG 對比測試 - 讓你直觀感受哪個更好！")
    print("=" * 80)
    
    # 1. 初始化系統
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
        persist_directory="./chroma_db_visual"
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
    
    # 初始化所有方法
    print("初始化所有 RAG 方法...")
    subquery_rag = SubQueryDecompositionRAG(
        rag_pipeline=rag_pipeline,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        enable_parallel=True
    )
    
    hyde_rag = HyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        hypothetical_length=200,
        temperature=0.7
    )
    
    hybrid_rag = HybridSubqueryHyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        hypothetical_length=200,
        temperature_subquery=0.3,
        temperature_hyde=0.7,
        enable_parallel=True
    )
    print("✅ 系統初始化完成！")
    
    # 2. 測試查詢
    test_query = "transformer architecture and attention mechanism"
    
    print("\n" + "=" * 80)
    print(f"📝 測試問題: '{test_query}'")
    print("=" * 80)
    
    methods_results = {}
    
    # === 方法 1: 正常 RAG ===
    print("\n" + "🔵" * 40)
    print("【方法 1】正常 RAG")
    print("🔵" * 40)
    try:
        normal_start = time.time()
        normal_docs = vector_retriever.retrieve(query=test_query, top_k=3)
        normal_retrieval_time = time.time() - normal_start
        
        print(f"\n📚 檢索到的文檔（前 3 個）:")
        for i, doc in enumerate(normal_docs, 1):
            score = doc.get('score', 0)
            title = doc['metadata'].get('title', 'N/A')
            print(f"\n  {i}. 📄 {title[:70]}")
            print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
            print(f"     內容預覽: {doc['content'][:200]}...")
        
        # 生成答案
        normal_context = formatter.format_context(normal_docs, document_type="paper")
        normal_prompt = formatter.create_prompt(test_query, normal_context, document_type="paper")
        normal_answer_start = time.time()
        normal_answer = llm.generate(prompt=normal_prompt, temperature=0.7, max_tokens=500)
        normal_answer_time = time.time() - normal_answer_start
        normal_total_time = time.time() - normal_start
        
        print(f"\n💬 生成的答案:")
        print("-" * 60)
        print(normal_answer[:600])
        if len(normal_answer) > 600:
            print("...")
        print("-" * 60)
        
        # 評估答案質量
        normal_quality = evaluate_answer_quality(normal_answer, test_query)
        normal_avg_score = sum(doc.get('score', 0) for doc in normal_docs) / len(normal_docs) if normal_docs else 0
        
        methods_results['正常 RAG'] = {
            'docs': normal_docs,
            'count': len(normal_docs),
            'avg_score': normal_avg_score,
            'answer': normal_answer,
            'time': normal_total_time,
            'quality': normal_quality
        }
        
    except Exception as e:
        print(f"❌ 出錯: {e}")
        import traceback
        traceback.print_exc()
        methods_results['正常 RAG'] = {'docs': [], 'count': 0, 'avg_score': 0, 'answer': '', 'time': 0, 'quality': {}}
    
    # === 方法 2: Sub-query RAG ===
    print("\n" + "🟢" * 40)
    print("【方法 2】Sub-query Decomposition RAG")
    print("🟢" * 40)
    try:
        subquery_result = subquery_rag.query(question=test_query, top_k=3, return_sub_queries=True)
        
        if subquery_result.get('sub_queries'):
            print(f"\n🔍 拆解的子問題:")
            for i, sq in enumerate(subquery_result['sub_queries'], 1):
                print(f"   {i}. {sq}")
        
        print(f"\n📚 檢索到的文檔（前 3 個）:")
        for i, doc in enumerate(subquery_result['results'], 1):
            score = doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0)))
            title = doc['metadata'].get('title', 'N/A')
            print(f"\n  {i}. 📄 {title[:70]}")
            print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
            print(f"     內容預覽: {doc['content'][:200]}...")
        
        # 生成答案
        subquery_context = formatter.format_context(subquery_result['results'], document_type="paper")
        subquery_prompt = formatter.create_prompt(test_query, subquery_context, document_type="paper")
        subquery_answer = llm.generate(prompt=subquery_prompt, temperature=0.7, max_tokens=500)
        
        print(f"\n💬 生成的答案:")
        print("-" * 60)
        print(subquery_answer[:600])
        if len(subquery_answer) > 600:
            print("...")
        print("-" * 60)
        
        # 評估答案質量
        subquery_quality = evaluate_answer_quality(subquery_answer, test_query)
        subquery_avg_score = sum(doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0))) 
                                 for doc in subquery_result['results']) / len(subquery_result['results']) if subquery_result['results'] else 0
        
        methods_results['Sub-query RAG'] = {
            'docs': subquery_result['results'],
            'count': subquery_result['total_docs_found'],
            'avg_score': subquery_avg_score,
            'answer': subquery_answer,
            'time': subquery_result['elapsed_time'],
            'quality': subquery_quality,
            'sub_queries': subquery_result.get('sub_queries', [])
        }
        
    except Exception as e:
        print(f"❌ 出錯: {e}")
        import traceback
        traceback.print_exc()
        methods_results['Sub-query RAG'] = {'docs': [], 'count': 0, 'avg_score': 0, 'answer': '', 'time': 0, 'quality': {}}
    
    # === 方法 3: HyDE RAG ===
    print("\n" + "🟡" * 40)
    print("【方法 3】HyDE RAG")
    print("🟡" * 40)
    try:
        hyde_result = hyde_rag.query(question=test_query, top_k=3, return_hypothetical=True)
        
        if hyde_result.get('hypothetical_document'):
            print(f"\n📝 生成的假設性文檔:")
            print("-" * 60)
            print(hyde_result['hypothetical_document'][:300])
            print("-" * 60)
        
        print(f"\n📚 檢索到的文檔（前 3 個）:")
        for i, doc in enumerate(hyde_result['results'], 1):
            score = doc.get('score', 0)
            title = doc['metadata'].get('title', 'N/A')
            print(f"\n  {i}. 📄 {title[:70]}")
            print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
            print(f"     內容預覽: {doc['content'][:200]}...")
        
        # 生成答案
        hyde_context = formatter.format_context(hyde_result['results'], document_type="paper")
        hyde_prompt = formatter.create_prompt(test_query, hyde_context, document_type="paper")
        hyde_answer = llm.generate(prompt=hyde_prompt, temperature=0.7, max_tokens=500)
        
        print(f"\n💬 生成的答案:")
        print("-" * 60)
        print(hyde_answer[:600])
        if len(hyde_answer) > 600:
            print("...")
        print("-" * 60)
        
        # 評估答案質量
        hyde_quality = evaluate_answer_quality(hyde_answer, test_query)
        hyde_avg_score = sum(doc.get('score', 0) for doc in hyde_result['results']) / len(hyde_result['results']) if hyde_result['results'] else 0
        
        methods_results['HyDE RAG'] = {
            'docs': hyde_result['results'],
            'count': hyde_result['total_docs_found'],
            'avg_score': hyde_avg_score,
            'answer': hyde_answer,
            'time': hyde_result['elapsed_time'],
            'quality': hyde_quality,
            'hypothetical': hyde_result.get('hypothetical_document', '')
        }
        
    except Exception as e:
        print(f"❌ 出錯: {e}")
        import traceback
        traceback.print_exc()
        methods_results['HyDE RAG'] = {'docs': [], 'count': 0, 'avg_score': 0, 'answer': '', 'time': 0, 'quality': {}}
    
    # === 方法 4: Hybrid RAG ===
    print("\n" + "🟣" * 40)
    print("【方法 4】Hybrid (Sub-query + HyDE) RAG")
    print("🟣" * 40)
    try:
        hybrid_result = hybrid_rag.query(
            question=test_query, 
            top_k=3, 
            return_sub_queries=True,
            return_hypothetical=True
        )
        
        if hybrid_result.get('sub_queries'):
            print(f"\n🔍 拆解的子問題:")
            for i, sq in enumerate(hybrid_result['sub_queries'], 1):
                print(f"   {i}. {sq}")
        
        if hybrid_result.get('hypothetical_documents'):
            print(f"\n📝 為每個子問題生成的假設性文檔（示例）:")
            for sq, hypo_doc in list(hybrid_result['hypothetical_documents'].items())[:1]:
                print(f"   子問題: {sq}")
                print(f"   假設性文檔: {hypo_doc[:200]}...")
        
        print(f"\n📚 檢索到的文檔（前 3 個）:")
        for i, doc in enumerate(hybrid_result['results'], 1):
            score = doc.get('score', 0)
            title = doc['metadata'].get('title', 'N/A')
            print(f"\n  {i}. 📄 {title[:70]}")
            print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
            print(f"     內容預覽: {doc['content'][:200]}...")
        
        # 生成答案
        hybrid_context = formatter.format_context(hybrid_result['results'], document_type="paper")
        hybrid_prompt = formatter.create_prompt(test_query, hybrid_context, document_type="paper")
        hybrid_answer = llm.generate(prompt=hybrid_prompt, temperature=0.7, max_tokens=500)
        
        print(f"\n💬 生成的答案:")
        print("-" * 60)
        print(hybrid_answer[:600])
        if len(hybrid_answer) > 600:
            print("...")
        print("-" * 60)
        
        # 評估答案質量
        hybrid_quality = evaluate_answer_quality(hybrid_answer, test_query)
        hybrid_avg_score = sum(doc.get('score', 0) for doc in hybrid_result['results']) / len(hybrid_result['results']) if hybrid_result['results'] else 0
        
        methods_results['Hybrid RAG'] = {
            'docs': hybrid_result['results'],
            'count': hybrid_result['total_docs_found'],
            'avg_score': hybrid_avg_score,
            'answer': hybrid_answer,
            'time': hybrid_result['elapsed_time'],
            'quality': hybrid_quality,
            'sub_queries': hybrid_result.get('sub_queries', []),
            'hypothetical': hybrid_result.get('hypothetical_documents', {})
        }
        
    except Exception as e:
        print(f"❌ 出錯: {e}")
        import traceback
        traceback.print_exc()
        methods_results['Hybrid RAG'] = {'docs': [], 'count': 0, 'avg_score': 0, 'answer': '', 'time': 0, 'quality': {}}
    
    # === 綜合對比總結 ===
    print("\n" + "=" * 80)
    print("📊 綜合對比總結")
    print("=" * 80)
    
    # 1. 性能對比表
    print(f"\n📈 性能對比表:")
    print(f"{'方法':<25} {'文檔數':<10} {'平均分數':<12} {'答案長度':<12} {'耗時':<10}")
    print("-" * 80)
    for method_name, result in methods_results.items():
        print(f"{method_name:<25} {result['count']:<10} {result['avg_score']:<12.3f} "
              f"{len(result.get('answer', '')):<12} {result.get('time', 0):<10.2f}s")
    
    # 2. 答案質量評分
    print(f"\n⭐ 答案質量評分:")
    for method_name, result in methods_results.items():
        quality = result.get('quality', {})
        if quality:
            overall = quality.get('overall_score', 0)
            stars = "⭐" * int(overall * 10)  # 0-10 星
            print(f"   {method_name:<25} {stars} ({overall:.2f})")
            print(f"      - 關鍵詞覆蓋: {quality.get('keyword_coverage', 0):.1%}")
            print(f"      - 詳細程度: {quality.get('detail_score', 0):.1%}")
            print(f"      - 專業術語: {quality.get('tech_score', 0):.1%}")
    
    # 3. 關鍵詞匹配分析
    query_keywords = set(test_query.lower().split())
    print(f"\n🔑 關鍵詞匹配分析（問題關鍵詞: {', '.join(query_keywords)}）:")
    for method_name, result in methods_results.items():
        answer = result.get('answer', '')
        if answer:
            answer_lower = answer.lower()
            matched_keywords = [kw for kw in query_keywords if kw in answer_lower]
            match_rate = len(matched_keywords) / len(query_keywords) * 100 if query_keywords else 0
            bars = "█" * int(match_rate / 10)  # 每 10% 一個方塊
            print(f"   {method_name:<25} {bars} {len(matched_keywords)}/{len(query_keywords)} ({match_rate:.0f}%)")
    
    # 4. 文檔相關性視覺化
    print(f"\n⭐ 文檔相關性對比（星級越高越相關）:")
    for method_name, result in methods_results.items():
        avg_score = result.get('avg_score', 0)
        stars = "⭐" * int(avg_score * 10)  # 0-10 星
        print(f"   {method_name:<25} {stars} ({avg_score:.3f})")
    
    # 5. 答案詳細程度對比
    print(f"\n📏 答案詳細程度對比:")
    max_length = max(len(r.get('answer', '')) for r in methods_results.values()) if methods_results else 1
    for method_name, result in methods_results.items():
        length = len(result.get('answer', ''))
        bars = "█" * int((length / max_length) * 30) if max_length > 0 else ""  # 最多 30 個方塊
        print(f"   {method_name:<25} {bars} ({length} 字符)")
    
    # 6. 最終建議
    print("\n" + "=" * 80)
    print("💡 如何判斷哪個最好？")
    print("=" * 80)
    print("""
    📚 看文檔相關性：
       - 檢查每個方法找到的文檔標題是否真的與問題相關
       - 查看文檔內容預覽，看是否包含問題的關鍵信息
       - 星級越高，文檔越相關
    
    💬 看答案質量：
       - 哪個答案更準確地回答了問題？
       - 哪個答案更詳細、更完整？
       - 哪個答案包含更多專業術語和細節？
       - 關鍵詞匹配率越高越好
    
    ⏱️ 看響應時間：
       - 如果質量相近，選擇更快的
       - 如果質量差異大，優先選擇質量好的
    
    🏆 綜合建議：
    """)
    
    # 找出最佳方法
    best_quality = None
    best_quality_method = None
    best_score = None
    best_score_method = None
    
    for method_name, result in methods_results.items():
        quality = result.get('quality', {})
        if quality:
            overall = quality.get('overall_score', 0)
            if best_quality is None or overall > best_quality:
                best_quality = overall
                best_quality_method = method_name
        
        avg_score = result.get('avg_score', 0)
        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_score_method = method_name
    
    if best_quality_method:
        print(f"   ✅ 答案質量最佳: {best_quality_method} (質量分數: {best_quality:.2f})")
    if best_score_method:
        print(f"   ✅ 文檔相關性最佳: {best_score_method} (平均分數: {best_score:.3f})")
    
    print("\n" + "=" * 80)


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="測試 HyDE RAG")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="執行對比測試（HyDE vs 正常 RAG）"
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="執行基本功能測試"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="執行融合方法對比測試（Hybrid vs 所有方法）"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="執行視覺化對比測試（顯示實際內容和答案，最直觀）"
    )
    
    args = parser.parse_args()
    
    if args.visual:
        test_visual_comparison_with_answers()
    elif args.hybrid:
        test_hybrid_vs_all_methods()
    elif args.compare:
        test_hyde_vs_normal_rag()
    elif args.basic:
        test_hyde_basic()
    else:
        # 預設執行視覺化測試（最直觀）
        test_visual_comparison_with_answers()


if __name__ == "__main__":
    main()

