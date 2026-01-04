"""
全面 RAG 測試腳本：測試所有基礎和進階 RAG 方法
包括：正常 RAG、SubQuery、HyDE、Step-back、Hybrid Subquery+HyDE、Triple Hybrid
"""
import os
import sys
import time
from typing import Dict, List, Optional
from src import (
    DocumentProcessor,
    BM25Retriever,
    VectorRetriever,
    HybridSearch,
    Reranker,
    RAGPipeline,
    PromptFormatter,
    OllamaLLM,
    SubQueryDecompositionRAG,
    HyDERAG,
    StepBackRAG,
    HybridSubqueryHyDERAG,
    TripleHybridRAG,
)


def initialize_system():
    """初始化所有系統組件"""
    print("\n" + "=" * 80)
    print("初始化系統組件")
    print("=" * 80)
    
    # 1. 處理文檔
    print("\n[1/5] 處理文檔...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    print("從 arXiv 獲取論文...")
    papers = processor.fetch_papers(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        max_results=20
    )
    print(f"✅ 獲取了 {len(papers)} 篇論文")
    
    documents = processor.process_documents(papers)
    print(f"✅ 總共創建了 {len(documents)} 個文檔 chunks")
    
    # 2. 初始化檢索器
    print("\n[2/5] 初始化檢索器...")
    bm25_retriever = BM25Retriever(documents)
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db_comprehensive"
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
    
    # 3. 初始化 LLM 和格式化器
    print("\n[3/5] 初始化 LLM 和格式化器...")
    try:
        llm = OllamaLLM(model_name="llama3.2:3b", timeout=180)
        print(f"✅ LLM 初始化完成: {llm.model_name}")
    except Exception as e:
        print(f"⚠️  LLM 初始化失敗: {e}")
        print("請確保 Ollama 正在運行並已下載模型")
        return None
    
    formatter = PromptFormatter(
        include_metadata=True,
        format_style="detailed"
    )
    
    # 4. 初始化所有 RAG 方法
    print("\n[4/5] 初始化所有 RAG 方法...")
    
    # 基礎方法
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
    
    step_back_rag = StepBackRAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        step_back_temperature=0.3,
        answer_temperature=0.7,
        enable_parallel=True
    )
    
    # 進階方法
    hybrid_subquery_hyde_rag = HybridSubqueryHyDERAG(
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
    
    triple_hybrid_rag = TripleHybridRAG(
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
    
    print("✅ 所有 RAG 方法初始化完成！")
    
    # 5. 返回所有組件
    print("\n[5/5] 系統初始化完成！")
    print("=" * 80)
    
    return {
        'vector_retriever': vector_retriever,
        'rag_pipeline': rag_pipeline,
        'llm': llm,
        'formatter': formatter,
        'subquery_rag': subquery_rag,
        'hyde_rag': hyde_rag,
        'step_back_rag': step_back_rag,
        'hybrid_subquery_hyde_rag': hybrid_subquery_hyde_rag,
        'triple_hybrid_rag': triple_hybrid_rag,
    }


def run_normal_rag(query: str, vector_retriever, llm, formatter) -> Dict:
    """運行正常 RAG（基礎方法）"""
    start_time = time.time()
    
    # 直接檢索
    results = vector_retriever.retrieve(query=query, top_k=5)
    
    # 格式化並生成答案
    context = formatter.format_context(results, document_type="paper")
    prompt = formatter.create_prompt(query, context, document_type="paper")
    
    answer_start = time.time()
    answer = llm.generate(prompt=prompt, temperature=0.7, max_tokens=2048)
    answer_time = time.time() - answer_start
    
    total_time = time.time() - start_time
    
    return {
        'method': '正常 RAG',
        'answer': answer,
        'results': results,
        'retrieval_time': total_time - answer_time,
        'answer_time': answer_time,
        'total_time': total_time,
        'docs_found': len(results)
    }


def test_all_basic_methods():
    """測試所有基礎 RAG 方法"""
    print("\n" + "=" * 80)
    print("【第一部分】基礎 RAG 方法測試")
    print("=" * 80)
    
    components = initialize_system()
    if not components:
        return
    
    test_queries = [
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
        "深度學習中的反向傳播算法原理",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: '{query}'")
        print("=" * 80)
        
        results = {}
        
        # 1. 正常 RAG
        print("\n[1] 正常 RAG")
        print("-" * 80)
        try:
            result = run_normal_rag(
                query,
                components['vector_retriever'],
                components['llm'],
                components['formatter']
            )
            results['normal'] = result
            print(f"✅ 完成 - 總耗時: {result['total_time']:.2f}s, 文檔數: {result['docs_found']}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['normal'] = None
        
        # 2. SubQuery RAG
        print("\n[2] SubQuery RAG")
        print("-" * 80)
        try:
            start_time = time.time()
            result = components['subquery_rag'].generate_answer(
                question=query,
                formatter=components['formatter'],
                top_k=5,
                document_type="paper",
                return_sub_queries=True
            )
            total_time = time.time() - start_time
            results['subquery'] = {
                'method': 'SubQuery RAG',
                'answer': result.get('answer', ''),
                'result': result,
                'total_time': total_time,
                'docs_found': result.get('total_docs_found', 0)
            }
            print(f"✅ 完成 - 總耗時: {total_time:.2f}s, 文檔數: {results['subquery']['docs_found']}")
            if result.get('sub_queries'):
                print(f"   子問題: {len(result['sub_queries'])} 個")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['subquery'] = None
        
        # 3. HyDE RAG
        print("\n[3] HyDE RAG")
        print("-" * 80)
        try:
            start_time = time.time()
            result = components['hyde_rag'].generate_answer(
                question=query,
                formatter=components['formatter'],
                top_k=5,
                document_type="paper",
                return_hypothetical=True
            )
            total_time = time.time() - start_time
            results['hyde'] = {
                'method': 'HyDE RAG',
                'answer': result.get('answer', ''),
                'result': result,
                'total_time': total_time,
                'docs_found': result.get('total_docs_found', 0)
            }
            print(f"✅ 完成 - 總耗時: {total_time:.2f}s, 文檔數: {results['hyde']['docs_found']}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['hyde'] = None
        
        # 4. Step-back RAG
        print("\n[4] Step-back RAG")
        print("-" * 80)
        try:
            start_time = time.time()
            result = components['step_back_rag'].generate_answer(
                question=query,
                formatter=components['formatter'],
                top_k=5,
                document_type="paper",
                return_abstract_question=True
            )
            total_time = time.time() - start_time
            results['step_back'] = {
                'method': 'Step-back RAG',
                'answer': result.get('answer', ''),
                'result': result,
                'total_time': total_time,
                'docs_found': len(result.get('specific_context', [])) + len(result.get('abstract_context', []))
            }
            print(f"✅ 完成 - 總耗時: {total_time:.2f}s, 文檔數: {results['step_back']['docs_found']}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['step_back'] = None
        
        # 顯示所有答案
        print("\n" + "=" * 80)
        print("【基礎方法答案對比】")
        print("=" * 80)
        
        for key, result in results.items():
            if result and result.get('answer'):
                print(f"\n{'=' * 80}")
                print(f"【{result['method']}】")
                print(f"{'=' * 80}")
                answer = result['answer']
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符, 耗時: {result['total_time']:.2f}s)")
                print(f"{'=' * 80}")


def test_all_advanced_methods():
    """測試所有進階 RAG 方法"""
    print("\n" + "=" * 80)
    print("【第二部分】進階 RAG 方法測試")
    print("=" * 80)
    
    components = initialize_system()
    if not components:
        return
    
    test_queries = [
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
        "深度學習中的反向傳播算法原理",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: '{query}'")
        print("=" * 80)
        
        results = {}
        
        # 1. Hybrid Subquery + HyDE RAG
        print("\n[1] Hybrid Subquery + HyDE RAG")
        print("-" * 80)
        try:
            start_time = time.time()
            result = components['hybrid_subquery_hyde_rag'].generate_answer(
                question=query,
                formatter=components['formatter'],
                top_k=5,
                document_type="paper",
                return_sub_queries=True,
                return_hypothetical=True
            )
            total_time = time.time() - start_time
            results['hybrid_subquery_hyde'] = {
                'method': 'Hybrid Subquery + HyDE RAG',
                'answer': result.get('answer', ''),
                'result': result,
                'total_time': total_time,
                'docs_found': result.get('total_docs_found', 0)
            }
            print(f"✅ 完成 - 總耗時: {total_time:.2f}s, 文檔數: {results['hybrid_subquery_hyde']['docs_found']}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['hybrid_subquery_hyde'] = None
        
        # 2. Triple Hybrid RAG
        print("\n[2] Triple Hybrid RAG (SubQuery + HyDE + Step-back)")
        print("-" * 80)
        try:
            start_time = time.time()
            result = components['triple_hybrid_rag'].generate_answer(
                question=query,
                formatter=components['formatter'],
                top_k=5,
                document_type="paper",
                return_sub_queries=True,
                return_hypothetical=True,
                return_abstract_question=True
            )
            total_time = time.time() - start_time
            results['triple_hybrid'] = {
                'method': 'Triple Hybrid RAG',
                'answer': result.get('answer', ''),
                'result': result,
                'total_time': total_time,
                'docs_found': result.get('total_docs_found', 0)
            }
            print(f"✅ 完成 - 總耗時: {total_time:.2f}s, 文檔數: {results['triple_hybrid']['docs_found']}")
        except Exception as e:
            print(f"❌ 出錯: {e}")
            results['triple_hybrid'] = None
        
        # 顯示所有答案
        print("\n" + "=" * 80)
        print("【進階方法答案對比】")
        print("=" * 80)
        
        for key, result in results.items():
            if result and result.get('answer'):
                print(f"\n{'=' * 80}")
                print(f"【{result['method']}】")
                print(f"{'=' * 80}")
                answer = result['answer']
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符, 耗時: {result['total_time']:.2f}s)")
                print(f"{'=' * 80}")


def test_pairwise_comparison():
    """兩兩對比測試（三組）"""
    print("\n" + "=" * 80)
    print("【第三部分】兩兩對比測試（三組）")
    print("=" * 80)
    
    components = initialize_system()
    if not components:
        return
    
    test_queries = [
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
    ]
    
    # 定義三組對比
    comparison_groups = [
        {
            'name': '組 1: SubQuery vs HyDE',
            'methods': [
                ('subquery', 'SubQuery RAG', components['subquery_rag']),
                ('hyde', 'HyDE RAG', components['hyde_rag'])
            ]
        },
        {
            'name': '組 2: HyDE vs Step-back',
            'methods': [
                ('hyde', 'HyDE RAG', components['hyde_rag']),
                ('step_back', 'Step-back RAG', components['step_back_rag'])
            ]
        },
        {
            'name': '組 3: Step-back vs Hybrid Subquery+HyDE',
            'methods': [
                ('step_back', 'Step-back RAG', components['step_back_rag']),
                ('hybrid_subquery_hyde', 'Hybrid Subquery+HyDE RAG', components['hybrid_subquery_hyde_rag'])
            ]
        }
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: '{query}'")
        print("=" * 80)
        
        for group in comparison_groups:
            print("\n" + "-" * 80)
            print(f"{group['name']}")
            print("-" * 80)
            
            results = {}
            
            for method_key, method_name, rag_instance in group['methods']:
                print(f"\n[{method_name}]")
                try:
                    start_time = time.time()
                    
                    if method_key == 'subquery':
                        result = rag_instance.generate_answer(
                            question=query,
                            formatter=components['formatter'],
                            top_k=5,
                            document_type="paper"
                        )
                    elif method_key == 'hyde':
                        result = rag_instance.generate_answer(
                            question=query,
                            formatter=components['formatter'],
                            top_k=5,
                            document_type="paper"
                        )
                    elif method_key == 'step_back':
                        result = rag_instance.generate_answer(
                            question=query,
                            formatter=components['formatter'],
                            top_k=5,
                            document_type="paper"
                        )
                    elif method_key == 'hybrid_subquery_hyde':
                        result = rag_instance.generate_answer(
                            question=query,
                            formatter=components['formatter'],
                            top_k=5,
                            document_type="paper"
                        )
                    
                    total_time = time.time() - start_time
                    results[method_key] = {
                        'method': method_name,
                        'answer': result.get('answer', ''),
                        'total_time': total_time,
                        'result': result
                    }
                    print(f"✅ 完成 - 耗時: {total_time:.2f}s")
                except Exception as e:
                    print(f"❌ 出錯: {e}")
                    results[method_key] = None
            
            # 顯示對比答案
            print(f"\n{'=' * 80}")
            print(f"【{group['name']} - 答案對比】")
            print(f"{'=' * 80}")
            
            for method_key, result in results.items():
                if result and result.get('answer'):
                    print(f"\n{'=' * 80}")
                    print(f"【{result['method']}】")
                    print(f"{'=' * 80}")
                    answer = result['answer']
                    for line in answer.strip().split('\n'):
                        if line.strip():
                            print(f"   {line}")
                        else:
                            print()
                    print(f"   (答案長度: {len(answer)} 字符, 耗時: {result['total_time']:.2f}s)")
                    print(f"{'=' * 80}")


def test_three_hybrid_comparison():
    """三個混合方法對比"""
    print("\n" + "=" * 80)
    print("【第四部分】三個混合方法對比")
    print("=" * 80)
    
    components = initialize_system()
    if not components:
        return
    
    test_queries = [
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
        "深度學習中的反向傳播算法原理",
    ]
    
    methods = [
        ('hybrid_subquery_hyde', 'Hybrid Subquery + HyDE RAG', components['hybrid_subquery_hyde_rag']),
        ('step_back', 'Step-back RAG', components['step_back_rag']),
        ('triple_hybrid', 'Triple Hybrid RAG', components['triple_hybrid_rag'])
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: '{query}'")
        print("=" * 80)
        
        results = {}
        
        for method_key, method_name, rag_instance in methods:
            print(f"\n[{method_name}]")
            print("-" * 80)
            try:
                start_time = time.time()
                
                if method_key == 'hybrid_subquery_hyde':
                    result = rag_instance.generate_answer(
                        question=query,
                        formatter=components['formatter'],
                        top_k=5,
                        document_type="paper"
                    )
                elif method_key == 'step_back':
                    result = rag_instance.generate_answer(
                        question=query,
                        formatter=components['formatter'],
                        top_k=5,
                        document_type="paper"
                    )
                elif method_key == 'triple_hybrid':
                    result = rag_instance.generate_answer(
                        question=query,
                        formatter=components['formatter'],
                        top_k=5,
                        document_type="paper"
                    )
                
                total_time = time.time() - start_time
                results[method_key] = {
                    'method': method_name,
                    'answer': result.get('answer', ''),
                    'total_time': total_time,
                    'result': result
                }
                print(f"✅ 完成 - 耗時: {total_time:.2f}s")
            except Exception as e:
                print(f"❌ 出錯: {e}")
                results[method_key] = None
        
        # 顯示對比答案
        print("\n" + "=" * 80)
        print("【三個混合方法答案對比】")
        print("=" * 80)
        
        for method_key, result in results.items():
            if result and result.get('answer'):
                print(f"\n{'=' * 80}")
                print(f"【{result['method']}】")
                print(f"{'=' * 80}")
                answer = result['answer']
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符, 耗時: {result['total_time']:.2f}s)")
                print(f"{'=' * 80}")


def test_all_methods_comprehensive():
    """全面對比：所有方法一起比較"""
    print("\n" + "=" * 80)
    print("【第五部分】全面對比 - 所有方法一起比較")
    print("=" * 80)
    
    components = initialize_system()
    if not components:
        return
    
    test_queries = [
        "什麼是區塊鏈的共識機制？",
        "transformer architecture and attention mechanism",
        "深度學習中的反向傳播算法原理",
    ]
    
    # 定義所有方法
    all_methods = [
        ('normal', '正常 RAG', None),
        ('subquery', 'SubQuery RAG', components['subquery_rag']),
        ('hyde', 'HyDE RAG', components['hyde_rag']),
        ('step_back', 'Step-back RAG', components['step_back_rag']),
        ('hybrid_subquery_hyde', 'Hybrid Subquery + HyDE RAG', components['hybrid_subquery_hyde_rag']),
        ('triple_hybrid', 'Triple Hybrid RAG', components['triple_hybrid_rag'])
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: '{query}'")
        print("=" * 80)
        
        results = {}
        
        # 運行所有方法
        for method_key, method_name, rag_instance in all_methods:
            print(f"\n[{method_name}]")
            print("-" * 80)
            try:
                start_time = time.time()
                
                if method_key == 'normal':
                    result = run_normal_rag(
                        query,
                        components['vector_retriever'],
                        components['llm'],
                        components['formatter']
                    )
                    results[method_key] = {
                        'method': method_name,
                        'answer': result['answer'],
                        'total_time': result['total_time'],
                        'docs_found': result['docs_found']
                    }
                else:
                    # 所有 RAG 方法都使用相同的 generate_answer 接口
                    result = rag_instance.generate_answer(
                        question=query,
                        formatter=components['formatter'],
                        top_k=5,
                        document_type="paper"
                    )
                    
                    total_time = time.time() - start_time
                    # 計算文檔數（不同方法可能有不同的結構）
                    docs_found = 0
                    if 'total_docs_found' in result:
                        docs_found = result['total_docs_found']
                    elif 'specific_context' in result and 'abstract_context' in result:
                        docs_found = len(result.get('specific_context', [])) + len(result.get('abstract_context', []))
                    elif 'results' in result:
                        docs_found = len(result.get('results', []))
                    
                    results[method_key] = {
                        'method': method_name,
                        'answer': result.get('answer', ''),
                        'total_time': total_time,
                        'docs_found': docs_found
                    }
                
                print(f"✅ 完成 - 耗時: {results[method_key]['total_time']:.2f}s, 文檔數: {results[method_key].get('docs_found', 0)}")
            except Exception as e:
                print(f"❌ 出錯: {e}")
                results[method_key] = None
        
        # 顯示性能統計
        print("\n" + "=" * 80)
        print("【性能統計】")
        print("=" * 80)
        print(f"{'方法':<40} {'耗時(秒)':<15} {'文檔數':<10}")
        print("-" * 80)
        for method_key, result in results.items():
            if result:
                print(f"{result['method']:<40} {result['total_time']:<15.2f} {result.get('docs_found', 0):<10}")
        
        # 顯示所有答案
        print("\n" + "=" * 80)
        print("【所有方法答案對比】")
        print("=" * 80)
        
        for method_key, result in results.items():
            if result and result.get('answer'):
                print(f"\n{'=' * 80}")
                print(f"【{result['method']}】")
                print(f"{'=' * 80}")
                answer = result['answer']
                for line in answer.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
                    else:
                        print()
                print(f"   (答案長度: {len(answer)} 字符, 耗時: {result['total_time']:.2f}s, 文檔數: {result.get('docs_found', 0)})")
                print(f"{'=' * 80}")
        
        # 評估提示
        print("\n" + "=" * 80)
        print("【答案質量評估提示】")
        print("=" * 80)
        print("請比較所有方法的答案，評估：")
        print("1. 哪個答案最全面（涵蓋更多方面）？")
        print("2. 哪個答案最淺顯易懂？")
        print("3. 哪個答案最專業準確？")
        print("4. 哪個答案結合了最多背景知識？")
        print("5. 哪個答案的邏輯最清晰？")
        print("6. 哪個方法在性能和質量之間取得最好平衡？")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="全面 RAG 測試")
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "advanced", "pairwise", "three_hybrid", "all", "comprehensive"],
        default="comprehensive",
        help="選擇測試類型: basic (基礎方法), advanced (進階方法), pairwise (兩兩對比), three_hybrid (三個混合), all (全部), comprehensive (全面對比)"
    )
    
    args = parser.parse_args()
    
    if args.test == "basic":
        test_all_basic_methods()
    elif args.test == "advanced":
        test_all_advanced_methods()
    elif args.test == "pairwise":
        test_pairwise_comparison()
    elif args.test == "three_hybrid":
        test_three_hybrid_comparison()
    elif args.test == "all":
        test_all_basic_methods()
        test_all_advanced_methods()
        test_pairwise_comparison()
        test_three_hybrid_comparison()
    elif args.test == "comprehensive":
        test_all_methods_comprehensive()

