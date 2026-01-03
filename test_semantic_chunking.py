"""
測試語義分塊功能的腳本

這個腳本演示如何：
1. 使用語義分塊處理私有檔案（PDF, DOCX, TXT）
2. 對比語義分塊 vs 字符分塊的效果
3. 建立 RAG 系統並測試檢索效果
4. 對比測試：有 RAG vs 無 RAG 的效果
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


def compare_chunking_methods(file_path: str):
    """
    對比語義分塊和字符分塊的效果
    
    Args:
        file_path: 要測試的檔案路徑
    """
    print("\n" + "=" * 60)
    print("對比：語義分塊 vs 字符分塊")
    print("=" * 60)
    
    # 初始化共用的 Embedding 模型
    print("\n[初始化] 載入 Embedding 模型...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from src.retrievers.vector_retriever import get_device
        
        hf_cache_dir = os.getenv("HF_CACHE_DIR", None)
        device = get_device()
        
        device_name_map = {
            'mps': 'MPS (macOS GPU)',
            'cuda': 'CUDA (NVIDIA GPU)',
            'cpu': 'CPU'
        }
        print(f"  使用設備: {device_name_map.get(device, device)}")
        
        model_kwargs = {'device': device}
        if hf_cache_dir:
            model_kwargs['cache_dir'] = hf_cache_dir
        
        shared_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        )
        print("  ✓ Embedding 模型載入完成")
    except Exception as e:
        print(f"  ❌ 載入 Embedding 模型失敗: {e}")
        print("  將只測試字符分塊模式")
        shared_embeddings = None
    
    # 1. 字符分塊
    print("\n[方法 1] 字符分塊（固定大小）")
    print("-" * 60)
    try:
        processor_char = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        documents_char = processor_char.process_file(str(file_path))
        print(f"  ✓ 創建了 {len(documents_char)} 個 chunks")
        
        if documents_char:
            print(f"\n  範例 chunk（第一個）：")
            print(f"    長度: {len(documents_char[0]['content'])} 字符")
            print(f"    內容預覽: {documents_char[0]['content'][:150]}...")
            
            # 統計資訊
            chunk_sizes = [len(doc['content']) for doc in documents_char]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            print(f"\n  統計資訊：")
            print(f"    平均大小: {avg_size:.0f} 字符")
            print(f"    最小大小: {min_size} 字符")
            print(f"    最大大小: {max_size} 字符")
    except Exception as e:
        print(f"  ❌ 字符分塊失敗: {e}")
        documents_char = None
    
    # 2. 語義分塊
    print("\n[方法 2] 語義分塊（基於語義相似度）")
    print("-" * 60)
    documents_semantic = None
    
    if shared_embeddings:
        try:
            processor_semantic = DocumentProcessor(
                embeddings=shared_embeddings,
                use_semantic_chunking=True,
                breakpoint_threshold_amount=1.5,
                min_chunk_size=100
            )
            print("  ⚠️  語義分塊需要計算 embedding，可能需要較長時間，請稍候...")
            documents_semantic = processor_semantic.process_file(str(file_path))
            print(f"  ✓ 創建了 {len(documents_semantic)} 個 chunks")
            
            if documents_semantic:
                print(f"\n  範例 chunk（第一個）：")
                print(f"    長度: {len(documents_semantic[0]['content'])} 字符")
                print(f"    內容預覽: {documents_semantic[0]['content'][:150]}...")
                
                # 統計資訊
                chunk_sizes = [len(doc['content']) for doc in documents_semantic]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)
                print(f"\n  統計資訊：")
                print(f"    平均大小: {avg_size:.0f} 字符")
                print(f"    最小大小: {min_size} 字符")
                print(f"    最大大小: {max_size} 字符")
        except ImportError as e:
            print(f"  ❌ 語義分塊需要安裝 langchain-experimental")
            print(f"    請執行: pip install langchain-experimental")
        except Exception as e:
            print(f"  ❌ 語義分塊失敗: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ⚠️  無法進行語義分塊（Embedding 模型載入失敗）")
    
    # 3. 對比總結
    print("\n" + "=" * 60)
    print("對比總結")
    print("=" * 60)
    
    if documents_char and documents_semantic:
        print(f"\n字符分塊: {len(documents_char)} 個 chunks")
        print(f"語義分塊: {len(documents_semantic)} 個 chunks")
        print(f"差異: {abs(len(documents_char) - len(documents_semantic))} 個 chunks")
        
        print("\n💡 觀察：")
        print("  - 語義分塊會根據語義邊界切分，不會在句子中間切斷")
        print("  - 字符分塊使用固定大小，可能會切斷句子")
        print("  - 語義分塊的 chunks 大小可能更不規則，但語義更完整")
    elif documents_char:
        print(f"\n字符分塊: {len(documents_char)} 個 chunks")
        print("語義分塊: 未完成（請檢查錯誤資訊）")
    
    return documents_char, documents_semantic, shared_embeddings


def test_with_semantic_chunking(
    file_path: str,
    test_query: str,
    use_semantic: bool = True
):
    """
    使用語義分塊（或字符分塊）建立 RAG 系統並測試
    
    Args:
        file_path: 檔案路徑
        test_query: 測試問題
        use_semantic: 是否使用語義分塊（True: 語義分塊, False: 字符分塊）
    """
    print("\n" + "=" * 60)
    print(f"使用 {'語義分塊' if use_semantic else '字符分塊'} 建立 RAG 系統")
    print("=" * 60)
    
    # 初始化共用的 Embedding 模型（如果需要語義分塊）
    shared_embeddings = None
    if use_semantic:
        print("\n[步驟 0] 初始化共用的 Embedding 模型...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from src.retrievers.vector_retriever import get_device
            
            hf_cache_dir = os.getenv("HF_CACHE_DIR", None)
            device = get_device()
            
            model_kwargs = {'device': device}
            if hf_cache_dir:
                model_kwargs['cache_dir'] = hf_cache_dir
            
            shared_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            print("  ✓ Embedding 模型初始化完成")
        except Exception as e:
            print(f"  ❌ 初始化失敗: {e}")
            print("  將回退到字符分塊模式")
            use_semantic = False
    
    # 1. 處理檔案
    print(f"\n[步驟 1] 處理檔案（使用{'語義分塊' if use_semantic else '字符分塊'}）...")
    print("-" * 60)
    
    try:
        if use_semantic and shared_embeddings:
            processor = DocumentProcessor(
                embeddings=shared_embeddings,
                use_semantic_chunking=True,
                breakpoint_threshold_amount=1.5,
                min_chunk_size=100
            )
            print("  ⚠️  語義分塊需要計算 embedding，可能需要較長時間，請稍候...")
        else:
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        documents = processor.process_file(str(file_path))
        print(f"  ✓ 處理完成，創建了 {len(documents)} 個 chunks")
        
        if documents:
            chunking_method = documents[0]['metadata'].get('chunking_method', 'character')
            print(f"  分塊方法: {chunking_method}")
            print(f"  範例 chunk 長度: {len(documents[0]['content'])} 字符")
    except ImportError as e:
        print(f"  ❌ 需要安裝 langchain-experimental: pip install langchain-experimental")
        return
    except Exception as e:
        print(f"  ❌ 處理檔案失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 初始化檢索系統
    print("\n[步驟 2] 初始化檢索系統...")
    print("-" * 60)
    
    try:
        print("  - 初始化 BM25 檢索器...")
        bm25_retriever = BM25Retriever(documents)
        
        print("  - 初始化向量檢索器...")
        # 使用不同的資料庫目錄避免衝突
        db_dir = "./chroma_db_semantic" if use_semantic else "./chroma_db_character"
        vector_retriever = VectorRetriever(
            documents,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory=db_dir,
            embeddings=shared_embeddings  # 傳入共用的 embeddings
        )
        
        print("  - 初始化混合搜尋...")
        hybrid_search = HybridSearch(
            sparse_retriever=bm25_retriever,
            dense_retriever=vector_retriever,
            fusion_method="rrf",
            rrf_k=60
        )
        
        print("  ✓ 檢索系統初始化完成")
    except Exception as e:
        print(f"  ❌ 檢索系統初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 初始化重排序和 RAG 管線
    print("\n[步驟 3] 初始化重排序和 RAG 管線...")
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
        
        print("  ✓ RAG 管線初始化完成")
    except Exception as e:
        print(f"  ❌ RAG 管線初始化失敗: {e}")
        print("   這可能是因為重排序模型下載失敗")
        print("   你可以繼續使用混合搜尋（不進行重排序）")
        return
    
    # 4. 初始化 LLM 和格式化器
    print("\n[步驟 4] 初始化 LLM 和格式化器...")
    print("-" * 60)
    
    try:
        print("  - 初始化 Prompt 格式化器...")
        formatter = PromptFormatter(format_style="detailed")
        
        print("  - 初始化 LLM...")
        llm = OllamaLLM(
            model_name="llama3.2:3b",
            timeout=180
        )
        
        print("  ✓ LLM 和格式化器初始化完成")
    except ConnectionError as e:
        print(f"  ❌ LLM 連接失敗: {e}")
        return
    except Exception as e:
        print(f"  ❌ 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 執行對比測試
    print("\n" + "=" * 60)
    print("開始執行 RAG 對比測試")
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


def main():
    """主函數"""
    print("=" * 60)
    print("語義分塊功能測試")
    print("=" * 60)
    print("\n這個腳本可以：")
    print("  1. 對比語義分塊和字符分塊的效果")
    print("  2. 使用語義分塊建立 RAG 系統並測試")
    print("  3. 對比有 RAG vs 無 RAG 的效果")
    
    # ========== 步驟 1: 準備檔案 ==========
    print("\n[步驟 1] 準備測試檔案")
    print("-" * 60)
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("\n請輸入檔案路徑（PDF, DOCX, 或 TXT）: ").strip()
        
        if not file_path:
            print("\n⚠️  未提供檔案路徑")
            print("\n使用方法：")
            print("  python test_semantic_chunking.py <檔案路徑> [測試問題]")
            print("\n範例：")
            print("  python test_semantic_chunking.py ./documents/my_document.pdf")
            return
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"\n❌ 檔案不存在: {file_path}")
        return
    
    print(f"✓ 找到檔案: {file_path}")
    print(f"  檔案類型: {file_path.suffix}")
    print(f"  檔案大小: {file_path.stat().st_size / 1024:.2f} KB")
    
    # ========== 步驟 2: 選擇測試模式 ==========
    print("\n[步驟 2] 選擇測試模式")
    print("-" * 60)
    print("\n請選擇測試模式：")
    print("  1. 對比語義分塊和字符分塊（只顯示分塊效果，不建立 RAG）")
    print("  2. 使用語義分塊建立 RAG 系統並測試")
    print("  3. 使用字符分塊建立 RAG 系統並測試（對比用）")
    print("  4. 全部執行（對比分塊 + 語義分塊 RAG + 字符分塊 RAG）")
    
    if len(sys.argv) > 2:
        # 從命令行參數獲取模式
        mode = sys.argv[2]
    else:
        mode = input("\n請輸入選項 (1/2/3/4，預設 2): ").strip() or "2"
    
    # ========== 步驟 3: 獲取測試問題 ==========
    test_query = None
    if mode in ["2", "3", "4"]:
        print("\n[步驟 3] 準備測試問題")
        print("-" * 60)
        
        if len(sys.argv) > 3:
            test_query = " ".join(sys.argv[3:])
        else:
            print("\n請輸入一個測試問題（應該涉及你的文檔內容）：")
            print("範例：")
            print("  - '這份文檔的主要內容是什麼？'")
            print("  - '文檔中提到了哪些關鍵概念？'")
            test_query = input("\n你的問題: ").strip()
        
        if not test_query:
            test_query = "這份文檔的主要內容是什麼？"
        
        print(f"✓ 測試問題: '{test_query}'")
    
    # ========== 執行測試 ==========
    if mode == "1":
        # 只對比分塊效果
        compare_chunking_methods(str(file_path))
    
    elif mode == "2":
        # 使用語義分塊建立 RAG
        test_with_semantic_chunking(
            file_path=str(file_path),
            test_query=test_query,
            use_semantic=True
        )
    
    elif mode == "3":
        # 使用字符分塊建立 RAG（對比用）
        test_with_semantic_chunking(
            file_path=str(file_path),
            test_query=test_query,
            use_semantic=False
        )
    
    elif mode == "4":
        # 全部執行
        print("\n" + "=" * 60)
        print("執行完整測試流程")
        print("=" * 60)
        
        # 1. 對比分塊
        documents_char, documents_semantic, shared_embeddings = compare_chunking_methods(str(file_path))
        
        # 2. 語義分塊 RAG
        if documents_semantic:
            print("\n\n" + "=" * 60)
            print("測試 1: 使用語義分塊的 RAG 系統")
            print("=" * 60)
            test_with_semantic_chunking(
                file_path=str(file_path),
                test_query=test_query,
                use_semantic=True
            )
        
        # 3. 字符分塊 RAG（對比）
        if documents_char:
            print("\n\n" + "=" * 60)
            print("測試 2: 使用字符分塊的 RAG 系統（對比用）")
            print("=" * 60)
            test_with_semantic_chunking(
                file_path=str(file_path),
                test_query=test_query,
                use_semantic=False
            )
    
    else:
        print(f"\n❌ 無效的選項: {mode}")
        print("請選擇 1, 2, 3, 或 4")
    
    print("\n" + "=" * 60)
    print("所有測試完成！")
    print("=" * 60)
    print("\n💡 提示：")
    print("  - 語義分塊能保持語義完整性，不會在句子中間切斷")
    print("  - 字符分塊速度更快，但可能會切斷句子")
    print("  - 可以對比兩種分塊方式的檢索效果，選擇最適合你的方式")


if __name__ == "__main__":
    main()

