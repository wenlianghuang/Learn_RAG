"""
Triple Hybrid RAG 測試腳本：測試融合 SubQuery + HyDE + Step-back 的三重混合 RAG
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
    TripleHybridRAG,
    StepBackRAG,
    HybridSubqueryHyDERAG,
)


def test_triple_hybrid_rag():
    """基本測試：三重混合 RAG"""
    print("=" * 60)
    print("Triple Hybrid RAG 基本測試")
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
        persist_directory="./chroma_db_triple"
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
    
    # 初始化 Triple Hybrid RAG
    print("初始化 Triple Hybrid RAG...")
    triple_rag = TripleHybridRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        hypothetical_length=200,
        temperature_subquery=0.3,
        temperature_hyde=0.7,
        temperature_stepback=0.3,
        answer_temperature=0.7,
        enable_parallel=True
    )
    print("✅ 系統初始化完成！")
    
    # 2. 測試查詢
    test_queries = [
        "2024年版MacBook Pro運行Python發燙問題",
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
        "How do neural networks learn and optimize?",
        "深度學習中的反向傳播算法原理",
    ]
    
    print("\n" + "=" * 60)
    print("開始測試")
    print("=" * 60)
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"測試查詢: '{query}'")
        print("=" * 60)
        
        try:
            start_time = time.time()
            result = triple_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_sub_queries=True,
                return_hypothetical=True,
                return_abstract_question=True
            )
            total_time = time.time() - start_time
            
            print(f"✅ Triple Hybrid RAG 完成")
            print(f"   檢索耗時: {result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {result['answer_time']:.2f}s")
            print(f"   總耗時: {total_time:.2f}s")
            print(f"   子問題檢索結果數: {len(result.get('subquery_results', []))}")
            print(f"   具體事實文檔數: {len(result.get('specific_context', []))}")
            print(f"   抽象原理文檔數: {len(result.get('abstract_context', []))}")
            print(f"   去重後總文檔數: {result['total_docs_found']}")
            
            if result.get('sub_queries'):
                print(f"\n   生成的子問題:")
                for i, sq in enumerate(result['sub_queries'], 1):
                    print(f"   {i}. {sq}")
            
            if result.get('abstract_question'):
                print(f"\n   生成的抽象問題:")
                print(f"   {result['abstract_question']}")
            
            # 顯示檢索結果
            print(f"\n   子問題檢索結果 (前 3 個):")
            for i, doc in enumerate(result.get('subquery_results', [])[:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            print(f"\n   具體事實文檔 (前 3 個):")
            for i, doc in enumerate(result.get('specific_context', [])[:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            print(f"\n   抽象原理文檔 (前 3 個):")
            for i, doc in enumerate(result.get('abstract_context', [])[:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            # 顯示完整答案
            print(f"\n   {'=' * 60}")
            print(f"   【Triple Hybrid RAG 生成的完整答案】")
            print(f"   {'=' * 60}")
            if result.get('answer'):
                answer = result['answer']
                answer_lines = answer.strip().split('\n')
                for line in answer_lines:
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"\n   (答案長度: {len(answer)} 字符)")
            else:
                print("   (未生成答案)")
            print(f"   {'=' * 60}")
            
        except Exception as e:
            print(f"❌ Triple Hybrid RAG 出錯: {e}")
            import traceback
            traceback.print_exc()


def test_triple_vs_others():
    """對比測試：Triple Hybrid RAG vs 其他方法"""
    print("=" * 60)
    print("Triple Hybrid RAG vs 其他方法對比測試")
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
    
    # 初始化所有 RAG 系統
    print("初始化 RAG 系統...")
    triple_rag = TripleHybridRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        enable_parallel=True
    )
    
    step_back_rag = StepBackRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        enable_parallel=True
    )
    
    hybrid_subquery_hyde_rag = HybridSubqueryHyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=5,
        enable_parallel=True
    )
    
    print("✅ 系統初始化完成！")
    
    # 2. 測試查詢
    test_queries = [
        "2024年版MacBook Pro運行Python發燙問題",
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
    ]
    
    print("\n" + "=" * 60)
    print("開始對比測試")
    print("=" * 60)
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"測試查詢: '{query}'")
        print("=" * 60)
        
        results = {}
        
        # === 方法 1: Step-back RAG ===
        print("\n[方法 1] Step-back RAG")
        print("-" * 60)
        try:
            start_time = time.time()
            step_back_result = step_back_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_abstract_question=True
            )
            step_back_time = time.time() - start_time
            
            print(f"✅ Step-back RAG 完成")
            print(f"   檢索耗時: {step_back_result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {step_back_result['answer_time']:.2f}s")
            print(f"   總耗時: {step_back_time:.2f}s")
            print(f"   具體事實: {len(step_back_result.get('specific_context', []))} 個")
            print(f"   抽象原理: {len(step_back_result.get('abstract_context', []))} 個")
            
            results['step_back'] = {
                'result': step_back_result,
                'time': step_back_time
            }
        except Exception as e:
            print(f"❌ Step-back RAG 出錯: {e}")
            results['step_back'] = None
        
        # === 方法 2: Hybrid Subquery + HyDE RAG ===
        print("\n[方法 2] Hybrid Subquery + HyDE RAG")
        print("-" * 60)
        try:
            start_time = time.time()
            hybrid_result = hybrid_subquery_hyde_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_sub_queries=True,
                return_hypothetical=True
            )
            hybrid_time = time.time() - start_time
            
            print(f"✅ Hybrid Subquery + HyDE RAG 完成")
            print(f"   檢索耗時: {hybrid_result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {hybrid_result['answer_time']:.2f}s")
            print(f"   總耗時: {hybrid_time:.2f}s")
            print(f"   子問題數: {len(hybrid_result.get('sub_queries', []))}")
            print(f"   總文檔數: {hybrid_result['total_docs_found']}")
            
            results['hybrid'] = {
                'result': hybrid_result,
                'time': hybrid_time
            }
        except Exception as e:
            print(f"❌ Hybrid Subquery + HyDE RAG 出錯: {e}")
            results['hybrid'] = None
        
        # === 方法 3: Triple Hybrid RAG ===
        print("\n[方法 3] Triple Hybrid RAG (SubQuery + HyDE + Step-back)")
        print("-" * 60)
        try:
            start_time = time.time()
            triple_result = triple_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_sub_queries=True,
                return_hypothetical=True,
                return_abstract_question=True
            )
            triple_time = time.time() - start_time
            
            print(f"✅ Triple Hybrid RAG 完成")
            print(f"   檢索耗時: {triple_result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {triple_result['answer_time']:.2f}s")
            print(f"   總耗時: {triple_time:.2f}s")
            print(f"   子問題檢索: {len(triple_result.get('subquery_results', []))} 個")
            print(f"   具體事實: {len(triple_result.get('specific_context', []))} 個")
            print(f"   抽象原理: {len(triple_result.get('abstract_context', []))} 個")
            print(f"   去重後總計: {triple_result['total_docs_found']} 個")
            
            results['triple'] = {
                'result': triple_result,
                'time': triple_time
            }
        except Exception as e:
            print(f"❌ Triple Hybrid RAG 出錯: {e}")
            results['triple'] = None
        
        # === 顯示完整答案對比 ===
        print("\n" + "=" * 60)
        print("【完整答案對比】")
        print("=" * 60)
        
        # Step-back RAG 答案
        if results.get('step_back') and results['step_back']['result']:
            print(f"\n   {'=' * 60}")
            print(f"   【Step-back RAG 答案】")
            print(f"   {'=' * 60}")
            answer = results['step_back']['result'].get('answer', '')
            if answer:
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符)")
            print(f"   {'=' * 60}")
        
        # Hybrid Subquery + HyDE RAG 答案
        if results.get('hybrid') and results['hybrid']['result']:
            print(f"\n   {'=' * 60}")
            print(f"   【Hybrid Subquery + HyDE RAG 答案】")
            print(f"   {'=' * 60}")
            answer = results['hybrid']['result'].get('answer', '')
            if answer:
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符)")
            print(f"   {'=' * 60}")
        
        # Triple Hybrid RAG 答案
        if results.get('triple') and results['triple']['result']:
            print(f"\n   {'=' * 60}")
            print(f"   【Triple Hybrid RAG 答案】")
            print(f"   {'=' * 60}")
            answer = results['triple']['result'].get('answer', '')
            if answer:
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符)")
            print(f"   {'=' * 60}")
        
        # === 對比總結 ===
        print("\n" + "=" * 60)
        print("【對比總結】")
        print("=" * 60)
        if results.get('step_back'):
            print(f"Step-back RAG 總耗時: {results['step_back']['time']:.2f}s")
        if results.get('hybrid'):
            print(f"Hybrid Subquery + HyDE RAG 總耗時: {results['hybrid']['time']:.2f}s")
        if results.get('triple'):
            print(f"Triple Hybrid RAG 總耗時: {results['triple']['time']:.2f}s")
        
        print("\n" + "=" * 60)
        print("【答案質量比較提示】")
        print("=" * 60)
        print("請比較三種方法的答案，評估：")
        print("1. 哪個答案更全面（涵蓋更多方面）？")
        print("2. 哪個答案更淺顯易懂？")
        print("3. 哪個答案更專業準確？")
        print("4. 哪個答案結合了更多背景知識？")
        print("5. 哪個答案的邏輯更清晰？")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Triple Hybrid RAG 測試")
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "compare"],
        default="basic",
        help="選擇測試類型: basic (基本測試), compare (與其他方法對比)"
    )
    
    args = parser.parse_args()
    
    if args.test == "basic":
        test_triple_hybrid_rag()
    elif args.test == "compare":
        test_triple_vs_others()

