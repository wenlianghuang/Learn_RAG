"""
Step-back RAG 測試腳本：測試 Step-back Prompting 雙軌 RAG
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
    StepBackRAG,
    HyDERAG,
)

def test_step_back_rag():
    """測試 Step-back RAG"""
    print("=" * 60)
    print("Step-back Prompting 雙軌 RAG 測試")
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
        persist_directory="./chroma_db_stepback"
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
    
    # 初始化 Step-back RAG
    print("初始化 Step-back RAG...")
    step_back_rag = StepBackRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        step_back_temperature=0.3,
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
        
        # === Step-back RAG ===
        print("\n[Step-back RAG]")
        print("-" * 60)
        
        try:
            step_back_start = time.time()
            result = step_back_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_abstract_question=True
            )
            step_back_total_time = time.time() - step_back_start
            
            print(f"✅ Step-back RAG 完成")
            print(f"   檢索耗時: {result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {result['answer_time']:.2f}s")
            print(f"   總耗時: {step_back_total_time:.2f}s")
            print(f"   具體事實文檔數: {len(result['specific_context'])}")
            print(f"   抽象原理文檔數: {len(result['abstract_context'])}")
            
            if result.get('abstract_question'):
                print(f"\n   生成的抽象問題:")
                print(f"   {result['abstract_question']}")
            
            # 顯示檢索結果
            print(f"\n   具體事實文檔 (前 3 個):")
            for i, doc in enumerate(result['specific_context'][:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            print(f"\n   抽象原理文檔 (前 3 個):")
            for i, doc in enumerate(result['abstract_context'][:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            # 顯示完整答案
            print(f"\n   {'=' * 60}")
            print(f"   【Step-back RAG 生成的完整答案】")
            print(f"   {'=' * 60}")
            if result.get('answer'):
                answer = result['answer']
                # 按行分割答案，每行前面加缩进
                answer_lines = answer.strip().split('\n')
                for line in answer_lines:
                    if line.strip():  # 跳过空行
                        print(f"   {line}")
                    else:
                        print()  # 保留空行
                print(f"\n   (答案長度: {len(answer)} 字符)")
            else:
                print("   (未生成答案)")
            print(f"   {'=' * 60}")
            
        except Exception as e:
            print(f"❌ Step-back RAG 出錯: {e}")
            import traceback
            traceback.print_exc()


def test_step_back_vs_normal_rag():
    """對比測試：Step-back RAG vs 正常 RAG"""
    print("=" * 60)
    print("Step-back RAG vs 正常 RAG 對比測試")
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
    
    # 初始化 Step-back RAG
    print("初始化 Step-back RAG...")
    step_back_rag = StepBackRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        step_back_temperature=0.3,
        answer_temperature=0.7,
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
            
            # 顯示完整答案
            print(f"\n   {'=' * 60}")
            print(f"   【正常 RAG 生成的完整答案】")
            print(f"   {'=' * 60}")
            if normal_answer:
                # 按行分割答案，每行前面加缩进
                answer_lines = normal_answer.strip().split('\n')
                for line in answer_lines:
                    if line.strip():  # 跳过空行
                        print(f"   {line}")
                    else:
                        print()  # 保留空行
                print(f"\n   (答案長度: {len(normal_answer)} 字符)")
            else:
                print("   (未生成答案)")
            print(f"   {'=' * 60}")
            
        except Exception as e:
            print(f"❌ 正常 RAG 出錯: {e}")
            normal_answer = None
            normal_total_time = 0
            normal_results = []
            import traceback
            traceback.print_exc()
        
        # === 方法 2: Step-back RAG（雙軌檢索）===
        print("\n[方法 2] Step-back RAG（雙軌檢索）")
        print("-" * 60)
        
        try:
            step_back_start = time.time()
            step_back_result = step_back_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_abstract_question=True
            )
            step_back_total_time = time.time() - step_back_start
            
            print(f"✅ Step-back RAG 完成")
            print(f"   抽象問題生成: {step_back_result['elapsed_time'] - (step_back_result.get('answer_time', 0)):.2f}s")
            print(f"   檢索耗時: {step_back_result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {step_back_result['answer_time']:.2f}s")
            print(f"   總耗時: {step_back_total_time:.2f}s")
            print(f"   具體事實文檔數: {len(step_back_result['specific_context'])}")
            print(f"   抽象原理文檔數: {len(step_back_result['abstract_context'])}")
            
            if step_back_result.get('abstract_question'):
                print(f"\n   生成的抽象問題:")
                print(f"   {step_back_result['abstract_question']}")
            
            # 顯示檢索結果
            print(f"\n   具體事實文檔 (前 3 個):")
            for i, doc in enumerate(step_back_result['specific_context'][:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            print(f"\n   抽象原理文檔 (前 3 個):")
            for i, doc in enumerate(step_back_result['abstract_context'][:3], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                if len(title) > 50:
                    title = title[:47] + "..."
                print(f"   {i}. {title} (分數: {score:.4f})")
            
            # 顯示完整答案
            print(f"\n   {'=' * 60}")
            print(f"   【Step-back RAG 生成的完整答案】")
            print(f"   {'=' * 60}")
            if step_back_result.get('answer'):
                answer = step_back_result['answer']
                # 按行分割答案，每行前面加缩进
                answer_lines = answer.strip().split('\n')
                for line in answer_lines:
                    if line.strip():  # 跳过空行
                        print(f"   {line}")
                    else:
                        print()  # 保留空行
                print(f"\n   (答案長度: {len(answer)} 字符)")
            else:
                print("   (未生成答案)")
            print(f"   {'=' * 60}")
            
        except Exception as e:
            print(f"❌ Step-back RAG 出錯: {e}")
            step_back_result = None
            step_back_total_time = 0
            import traceback
            traceback.print_exc()
        
        # === 對比總結 ===
        print("\n" + "=" * 60)
        print("【對比總結】")
        print("=" * 60)
        if normal_total_time > 0 and step_back_total_time > 0:
            print(f"正常 RAG 總耗時: {normal_total_time:.2f}s")
            print(f"Step-back RAG 總耗時: {step_back_total_time:.2f}s")
            print(f"時間差異: {abs(step_back_total_time - normal_total_time):.2f}s")
            if step_back_result:
                total_stepback_docs = len(step_back_result.get('specific_context', [])) + len(step_back_result.get('abstract_context', []))
                print(f"正常 RAG 檢索文檔數: {len(normal_results)}")
                print(f"Step-back RAG 檢索文檔數: {total_stepback_docs} (具體事實: {len(step_back_result.get('specific_context', []))}, 抽象原理: {len(step_back_result.get('abstract_context', []))})")
        
        print("\n" + "=" * 60)
        print("【答案質量比較提示】")
        print("=" * 60)
        print("請比較兩種方法的答案，評估：")
        print("1. 哪個答案更淺顯易懂？")
        print("2. 哪個答案更專業準確？")
        print("3. 哪個答案結合了更多背景知識？")
        print("4. 哪個答案的邏輯更清晰？")
        print("=" * 60)


def test_step_back_vs_hyde():
    """對比測試：Step-back RAG vs HyDE RAG"""
    print("=" * 60)
    print("Step-back RAG vs HyDE RAG 對比測試")
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
    
    # 初始化兩種 RAG
    print("初始化 RAG 系統...")
    step_back_rag = StepBackRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        step_back_temperature=0.3,
        answer_temperature=0.7,
        enable_parallel=True
    )
    
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
        
        # === HyDE RAG ===
        print("\n[方法 1] HyDE RAG")
        print("-" * 60)
        
        try:
            hyde_start = time.time()
            hyde_result = hyde_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_hypothetical=True
            )
            hyde_total_time = time.time() - hyde_start
            
            print(f"✅ HyDE RAG 完成")
            print(f"   假設性文檔生成: {hyde_result.get('hypothetical_time', 0):.2f}s")
            print(f"   檢索耗時: {hyde_result.get('retrieval_time', 0):.2f}s")
            print(f"   生成耗時: {hyde_result.get('answer_time', 0):.2f}s")
            print(f"   總耗時: {hyde_total_time:.2f}s")
            print(f"   找到文檔數: {hyde_result['total_docs_found']}")
            
            if hyde_result.get('hypothetical_document'):
                print(f"\n   生成的假設性文檔:")
                hypo_doc = hyde_result['hypothetical_document']
                print(f"   {hypo_doc[:200]}...")
            
        except Exception as e:
            print(f"❌ HyDE RAG 出錯: {e}")
            hyde_result = None
            hyde_total_time = 0
            import traceback
            traceback.print_exc()
        
        # === Step-back RAG ===
        print("\n[方法 2] Step-back RAG")
        print("-" * 60)
        
        try:
            step_back_start = time.time()
            step_back_result = step_back_rag.generate_answer(
                question=query,
                formatter=formatter,
                top_k=5,
                document_type="paper",
                return_abstract_question=True
            )
            step_back_total_time = time.time() - step_back_start
            
            print(f"✅ Step-back RAG 完成")
            print(f"   檢索耗時: {step_back_result['elapsed_time']:.2f}s")
            print(f"   生成耗時: {step_back_result['answer_time']:.2f}s")
            print(f"   總耗時: {step_back_total_time:.2f}s")
            print(f"   具體事實文檔數: {len(step_back_result['specific_context'])}")
            print(f"   抽象原理文檔數: {len(step_back_result['abstract_context'])}")
            
            if step_back_result.get('abstract_question'):
                print(f"\n   生成的抽象問題:")
                print(f"   {step_back_result['abstract_question']}")
            
        except Exception as e:
            print(f"❌ Step-back RAG 出錯: {e}")
            step_back_result = None
            step_back_total_time = 0
            import traceback
            traceback.print_exc()
        
        # === 對比總結 ===
        print("\n" + "-" * 60)
        print("對比總結")
        print("-" * 60)
        if hyde_total_time > 0 and step_back_total_time > 0:
            print(f"HyDE RAG 總耗時: {hyde_total_time:.2f}s")
            print(f"Step-back RAG 總耗時: {step_back_total_time:.2f}s")
            print(f"時間差異: {abs(step_back_total_time - hyde_total_time):.2f}s")
            if hyde_result and step_back_result:
                total_hyde_docs = hyde_result.get('total_docs_found', 0)
                total_stepback_docs = len(step_back_result.get('specific_context', [])) + len(step_back_result.get('abstract_context', []))
                print(f"HyDE RAG 檢索文檔數: {total_hyde_docs}")
                print(f"Step-back RAG 檢索文檔數: {total_stepback_docs}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step-back RAG 測試")
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "vs_normal", "vs_hyde"],
        default="basic",
        help="選擇測試類型: basic (基本測試), vs_normal (與正常 RAG 對比), vs_hyde (與 HyDE RAG 對比)"
    )
    
    args = parser.parse_args()
    
    if args.test == "basic":
        test_step_back_rag()
    elif args.test == "vs_normal":
        test_step_back_vs_normal_rag()
    elif args.test == "vs_hyde":
        test_step_back_vs_hyde()

