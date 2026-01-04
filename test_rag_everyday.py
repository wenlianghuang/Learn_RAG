"""
日常問題 RAG 對比測試：使用 Wikipedia 等大眾化資料來源
測試更簡單、一般人能理解的問題
"""
import os
import sys
import time
import hashlib
from typing import List, Dict
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

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("⚠️  未安裝 wikipedia 套件，將使用示例文檔")
    print("   安裝命令: pip install wikipedia")


def fetch_wikipedia_articles(titles: List[str]) -> List[Dict]:
    """
    從 Wikipedia 獲取文章
    
    Args:
        titles: 文章標題列表
        
    Returns:
        文章列表，每個包含標題和內容
    """
    articles = []
    
    if not WIKIPEDIA_AVAILABLE:
        # 如果沒有安裝 wikipedia，使用示例文檔
        print("⚠️  使用示例文檔（建議安裝 wikipedia 套件以獲取真實資料）")
        example_articles = [
            {
                "title": "人工智慧",
                "content": """人工智慧（Artificial Intelligence, AI）是電腦科學的一個分支，旨在創建能夠執行通常需要人類智能的任務的系統。
人工智慧包括機器學習、深度學習、自然語言處理等技術。機器學習是人工智慧的一個子領域，它使電腦能夠從數據中學習，而無需明確編程。
深度學習是機器學習的一個子集，使用神經網路來模擬人腦的工作方式。自然語言處理使電腦能夠理解和生成人類語言。
人工智慧在醫療、金融、交通、教育等領域都有廣泛應用。現代 AI 系統可以進行圖像識別、語音識別、自動駕駛、智能推薦等任務。
AI 技術正在改變我們的生活方式，從智能手機的語音助手到自動駕駛汽車，從醫療診斷到金融分析，無處不在。"""
            },
            {
                "title": "機器學習",
                "content": """機器學習（Machine Learning）是人工智慧的一個分支，專注於開發能夠從數據中學習的算法和統計模型。
機器學習算法通過分析大量數據來識別模式並做出預測或決策。主要類型包括監督學習、無監督學習和強化學習。
監督學習使用標記的數據來訓練模型，例如分類和回歸問題。常見應用包括垃圾郵件過濾、圖像分類、價格預測等。
無監督學習從未標記的數據中發現隱藏的模式，例如聚類分析。應用包括客戶分群、異常檢測等。
強化學習通過與環境互動來學習，通過獎勵和懲罰來改進行為。應用包括遊戲 AI、機器人控制等。
機器學習在推薦系統、搜索引擎、語音助手、自動駕駛等領域有廣泛應用。"""
            },
            {
                "title": "深度學習",
                "content": """深度學習（Deep Learning）是機器學習的一個子集，使用多層神經網路來學習數據的表示。
深度學習模型可以自動從原始數據中提取特徵，無需人工特徵工程。常見的深度學習架構包括卷積神經網路（CNN）、循環神經網路（RNN）和 Transformer。
卷積神經網路主要用於圖像處理和計算機視覺任務，如人臉識別、物體檢測等。
循環神經網路適合處理序列數據，如自然語言和時間序列，應用包括語音識別、文本生成等。
Transformer 架構在自然語言處理領域取得了重大突破，如 BERT 和 GPT 模型，能夠理解語言的語義和上下文。
深度學習在語音識別、圖像識別、自然語言處理、自動翻譯等領域表現出色。"""
            },
            {
                "title": "自然語言處理",
                "content": """自然語言處理（Natural Language Processing, NLP）是人工智慧和語言學的交叉領域，旨在使電腦能夠理解、解釋和生成人類語言。
NLP 的任務包括文本分類、情感分析、機器翻譯、問答系統、文本摘要等。現代 NLP 主要依賴深度學習和 Transformer 架構。
詞嵌入技術如 Word2Vec 和 GloVe 將詞語轉換為數值向量，使電腦能夠理解詞語的語義關係。
預訓練語言模型如 BERT 和 GPT 在各種 NLP 任務上取得了優異的表現，能夠理解語言的上下文和語義。
NLP 應用包括智能助手（如 Siri、Alexa）、搜索引擎、自動翻譯、內容推薦、情感分析等。
這些應用讓電腦能夠理解和處理人類語言，大大提升了人機互動的體驗。"""
            },
            {
                "title": "神經網路",
                "content": """神經網路（Neural Network）是受生物神經系統啟發的計算模型，由相互連接的節點（神經元）組成。
人工神經網路由輸入層、隱藏層和輸出層組成。每個連接都有權重，通過訓練過程調整這些權重來學習模式。
反向傳播算法是訓練神經網路的關鍵技術，通過計算梯度來更新權重，使網路能夠從錯誤中學習。
深度神經網路有多個隱藏層，能夠學習更複雜的模式和特徵。層數越多，網路越深，學習能力越強。
神經網路在圖像識別、語音識別、自然語言處理、遊戲 AI 等領域取得了重大成功。
這些網路能夠自動從數據中學習複雜的模式，無需人工設計特徵，這是它們強大的原因。"""
            }
        ]
        return example_articles
    
    # 設置 Wikipedia 語言（中文）
    try:
        wikipedia.set_lang("zh")
    except:
        pass
    
    for title in titles:
        try:
            print(f"  正在獲取: {title}...")
            page = wikipedia.page(title, auto_suggest=False)
            articles.append({
                "title": page.title,
                "content": page.content
            })
            print(f"  ✅ 成功獲取: {page.title} ({len(page.content)} 字符)")
        except wikipedia.exceptions.DisambiguationError as e:
            # 如果有歧義，使用第一個選項
            try:
                page = wikipedia.page(e.options[0])
                articles.append({
                    "title": page.title,
                    "content": page.content
                })
                print(f"  ✅ 成功獲取（歧義解決）: {page.title}")
            except:
                print(f"  ⚠️  無法獲取: {title}")
        except Exception as e:
            print(f"  ⚠️  無法獲取: {title} ({e})")
            continue
    
    return articles


def process_wikipedia_articles(articles: List[Dict], processor: DocumentProcessor) -> List[Dict]:
    """
    處理 Wikipedia 文章，轉換為文檔格式
    
    Args:
        articles: 文章列表
        processor: 文檔處理器
        
    Returns:
        處理後的文檔 chunks
    """
    documents = []
    
    for article in articles:
        # 使用標題和內容
        full_text = f"標題: {article['title']}\n\n內容: {article['content']}"
        
        # 分割文字
        chunks = processor.text_splitter.split_text(full_text)
        
        # 為每個 chunk 創建文檔物件
        for i, chunk in enumerate(chunks):
            doc = {
                "content": chunk,
                "metadata": {
                    "title": article['title'],
                    "source": "Wikipedia",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunking_method": "character"
                }
            }
            documents.append(doc)
    
    return documents


def evaluate_answer_quality(answer: str, query: str) -> dict:
    """評估答案質量"""
    query_keywords = set(query.lower().split())
    answer_lower = answer.lower()
    
    # 關鍵詞覆蓋率
    matched = sum(1 for kw in query_keywords if kw in answer_lower)
    keyword_coverage = matched / len(query_keywords) if query_keywords else 0
    
    # 答案詳細程度
    detail_score = min(len(answer) / 500, 1.0)
    
    # 專業術語數量（簡化版）
    technical_terms = ['方法', '技術', '應用', '原理', '系統', '算法', 
                      'method', 'technique', 'application', 'principle', 'system', 'algorithm']
    tech_count = sum(1 for term in technical_terms if term in answer_lower)
    tech_score = min(tech_count / 5, 1.0)
    
    overall_score = (keyword_coverage * 0.4 + detail_score * 0.3 + tech_score * 0.3)
    
    return {
        'keyword_coverage': keyword_coverage,
        'detail_score': detail_score,
        'tech_score': tech_score,
        'overall_score': overall_score
    }


def test_everyday_rag_comparison():
    """日常問題 RAG 對比測試"""
    print("=" * 80)
    print("🎯 日常問題 RAG 對比測試 - 使用大眾化資料來源")
    print("=" * 80)
    
    # 1. 獲取 Wikipedia 文章
    print("\n[1/6] 獲取 Wikipedia 文章...")
    article_titles = [
        "人工智慧",
        "機器學習",
        "深度學習",
        "自然語言處理",
        "神經網路"
    ]
    
    articles = fetch_wikipedia_articles(article_titles)
    print(f"✅ 獲取了 {len(articles)} 篇文章")
    
    # 2. 處理文檔
    print("\n[2/6] 處理文檔並分割成 chunks...")
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
    documents = process_wikipedia_articles(articles, processor)
    print(f"✅ 總共創建了 {len(documents)} 個文檔 chunks")
    
    # 3. 初始化檢索器
    print("\n[3/6] 初始化檢索器...")
    bm25_retriever = BM25Retriever(documents)
    vector_retriever = VectorRetriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db_everyday"
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
        recall_k=20,
        adaptive_recall=True
    )
    
    # 4. 初始化 LLM 和格式化器
    print("\n[4/6] 初始化 LLM 和格式化器...")
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
    
    # 5. 初始化所有 RAG 方法
    print("\n[5/6] 初始化所有 RAG 方法...")
    subquery_rag = SubQueryDecompositionRAG(
        rag_pipeline=rag_pipeline,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=4,
        enable_parallel=True
    )
    
    hyde_rag = HyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        hypothetical_length=150,
        temperature=0.7
    )
    
    hybrid_rag = HybridSubqueryHyDERAG(
        rag_pipeline=rag_pipeline,
        vector_retriever=vector_retriever,
        llm=llm,
        max_sub_queries=3,
        top_k_per_subquery=4,
        hypothetical_length=150,
        temperature_subquery=0.3,
        temperature_hyde=0.7,
        enable_parallel=True
    )
    print("✅ 系統初始化完成！")
    
    # 6. 測試查詢（簡單、日常的問題）
    print("\n[6/6] 開始測試...")
    print("=" * 80)
    
    # 簡單、一般人能理解的問題
    test_queries = [
        "什麼是人工智慧？它有哪些應用？",
        "機器學習和深度學習有什麼不同？",
        "自然語言處理可以用來做什麼？",
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"📝 測試問題: '{query}'")
        print("=" * 80)
        
        methods_results = {}
        
        # === 方法 1: 正常 RAG ===
        print("\n" + "🔵" * 40)
        print("【方法 1】正常 RAG")
        print("🔵" * 40)
        try:
            normal_start = time.time()
            normal_docs = vector_retriever.retrieve(query=query, top_k=3)
            normal_retrieval_time = time.time() - normal_start
            
            print(f"\n📚 檢索到的文檔（前 3 個）:")
            for i, doc in enumerate(normal_docs, 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                print(f"\n  {i}. 📄 {title}")
                print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
                print(f"     內容預覽: {doc['content'][:150]}...")
            
            # 生成答案
            normal_context = formatter.format_context(normal_docs, document_type="general")
            normal_prompt = formatter.create_prompt(query, normal_context, document_type="general")
            normal_answer_start = time.time()
            normal_answer = llm.generate(prompt=normal_prompt, temperature=0.7, max_tokens=400)
            normal_answer_time = time.time() - normal_answer_start
            normal_total_time = time.time() - normal_start
            
            print(f"\n💬 生成的答案:")
            print("-" * 60)
            print(normal_answer)
            print("-" * 60)
            
            normal_quality = evaluate_answer_quality(normal_answer, query)
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
            subquery_result = subquery_rag.query(question=query, top_k=3, return_sub_queries=True)
            
            if subquery_result.get('sub_queries'):
                print(f"\n🔍 拆解的子問題:")
                for i, sq in enumerate(subquery_result['sub_queries'], 1):
                    print(f"   {i}. {sq}")
            
            print(f"\n📚 檢索到的文檔（前 3 個）:")
            for i, doc in enumerate(subquery_result['results'], 1):
                score = doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0)))
                title = doc['metadata'].get('title', 'N/A')
                print(f"\n  {i}. 📄 {title}")
                print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
                print(f"     內容預覽: {doc['content'][:150]}...")
            
            # 生成答案
            subquery_context = formatter.format_context(subquery_result['results'], document_type="general")
            subquery_prompt = formatter.create_prompt(query, subquery_context, document_type="general")
            subquery_answer = llm.generate(prompt=subquery_prompt, temperature=0.7, max_tokens=400)
            
            print(f"\n💬 生成的答案:")
            print("-" * 60)
            print(subquery_answer)
            print("-" * 60)
            
            subquery_quality = evaluate_answer_quality(subquery_answer, query)
            subquery_avg_score = sum(doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0))) 
                                     for doc in subquery_result['results']) / len(subquery_result['results']) if subquery_result['results'] else 0
            
            methods_results['Sub-query RAG'] = {
                'docs': subquery_result['results'],
                'count': subquery_result['total_docs_found'],
                'avg_score': subquery_avg_score,
                'answer': subquery_answer,
                'time': subquery_result['elapsed_time'],
                'quality': subquery_quality
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
            hyde_result = hyde_rag.query(question=query, top_k=3, return_hypothetical=True)
            
            if hyde_result.get('hypothetical_document'):
                print(f"\n📝 生成的假設性文檔:")
                print("-" * 60)
                print(hyde_result['hypothetical_document'][:250])
                print("-" * 60)
            
            print(f"\n📚 檢索到的文檔（前 3 個）:")
            for i, doc in enumerate(hyde_result['results'], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                print(f"\n  {i}. 📄 {title}")
                print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
                print(f"     內容預覽: {doc['content'][:150]}...")
            
            # 生成答案
            hyde_context = formatter.format_context(hyde_result['results'], document_type="general")
            hyde_prompt = formatter.create_prompt(query, hyde_context, document_type="general")
            hyde_answer = llm.generate(prompt=hyde_prompt, temperature=0.7, max_tokens=400)
            
            print(f"\n💬 生成的答案:")
            print("-" * 60)
            print(hyde_answer)
            print("-" * 60)
            
            hyde_quality = evaluate_answer_quality(hyde_answer, query)
            hyde_avg_score = sum(doc.get('score', 0) for doc in hyde_result['results']) / len(hyde_result['results']) if hyde_result['results'] else 0
            
            methods_results['HyDE RAG'] = {
                'docs': hyde_result['results'],
                'count': hyde_result['total_docs_found'],
                'avg_score': hyde_avg_score,
                'answer': hyde_answer,
                'time': hyde_result['elapsed_time'],
                'quality': hyde_quality
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
                question=query, 
                top_k=3, 
                return_sub_queries=True,
                return_hypothetical=True
            )
            
            if hybrid_result.get('sub_queries'):
                print(f"\n🔍 拆解的子問題:")
                for i, sq in enumerate(hybrid_result['sub_queries'], 1):
                    print(f"   {i}. {sq}")
            
            print(f"\n📚 檢索到的文檔（前 3 個）:")
            for i, doc in enumerate(hybrid_result['results'], 1):
                score = doc.get('score', 0)
                title = doc['metadata'].get('title', 'N/A')
                print(f"\n  {i}. 📄 {title}")
                print(f"     相關性: {'⭐' * int(score * 5)} ({score:.3f})")
                print(f"     內容預覽: {doc['content'][:150]}...")
            
            # 生成答案
            hybrid_context = formatter.format_context(hybrid_result['results'], document_type="general")
            hybrid_prompt = formatter.create_prompt(query, hybrid_context, document_type="general")
            hybrid_answer = llm.generate(prompt=hybrid_prompt, temperature=0.7, max_tokens=400)
            
            print(f"\n💬 生成的答案:")
            print("-" * 60)
            print(hybrid_answer)
            print("-" * 60)
            
            hybrid_quality = evaluate_answer_quality(hybrid_answer, query)
            hybrid_avg_score = sum(doc.get('score', 0) for doc in hybrid_result['results']) / len(hybrid_result['results']) if hybrid_result['results'] else 0
            
            methods_results['Hybrid RAG'] = {
                'docs': hybrid_result['results'],
                'count': hybrid_result['total_docs_found'],
                'avg_score': hybrid_avg_score,
                'answer': hybrid_answer,
                'time': hybrid_result['elapsed_time'],
                'quality': hybrid_quality
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
        print(f"\n⭐ 答案質量評分（星級越高越好）:")
        for method_name, result in methods_results.items():
            quality = result.get('quality', {})
            if quality:
                overall = quality.get('overall_score', 0)
                stars = "⭐" * int(overall * 10)  # 0-10 星
                print(f"   {method_name:<25} {stars} ({overall:.2f})")
        
        # 3. 關鍵詞匹配分析
        query_keywords = set(query.lower().split())
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
        
        # 5. 最終建議
        print("\n" + "=" * 80)
        print("💡 如何判斷哪個最好？")
        print("=" * 80)
        print("""
    📚 看文檔相關性：
       - 檢查每個方法找到的文檔標題是否真的與問題相關
       - 查看文檔內容預覽，看是否包含問題的關鍵資訊
       - 星級越高，文檔越相關
    
    💬 看答案質量：
       - 哪個答案更準確地回答了問題？
       - 哪個答案更詳細、更完整？
       - 哪個答案更容易理解？
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
    
    parser = argparse.ArgumentParser(description="日常問題 RAG 對比測試")
    parser.add_argument(
        "--install-wikipedia",
        action="store_true",
        help="顯示安裝 Wikipedia 套件的命令"
    )
    
    args = parser.parse_args()
    
    if args.install_wikipedia:
        print("安裝 Wikipedia 套件:")
        print("  pip install wikipedia")
        print("\n或者使用 uv:")
        print("  uv pip install wikipedia")
        return
    
    test_everyday_rag_comparison()


if __name__ == "__main__":
    main()

