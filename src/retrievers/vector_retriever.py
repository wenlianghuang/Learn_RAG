"""
向量檢索器模組：使用 embedding 和向量資料庫進行語義檢索

支援兩種初始化方式：
1. 自動初始化 embeddings（預設）：根據參數創建新的 embedding 模型
2. 使用外部 embeddings：接收已初始化的 embedding 模型（可與 DocumentProcessor 共用）
"""
from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
from .base import BaseRetriever

# 嘗試導入 HuggingFaceEmbeddings（免費模型）
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("需要安裝 langchain-community 或 langchain-huggingface 才能使用 Hugging Face embeddings")

# 導入 torch 來檢測可用的設備
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> str:
    """
    自動檢測並返回最佳可用的設備
    
    Returns:
        設備名稱: 'mps' (macOS GPU), 'cuda' (NVIDIA GPU), 或 'cpu'
    """
    if not TORCH_AVAILABLE:
        return 'cpu'
    
    # 優先順序: MPS (macOS) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class VectorRetriever(BaseRetriever):
    """使用向量檢索進行語義搜尋"""
    
    def __init__(
        self,
        documents: List[Dict],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = "./chroma_db",
        hf_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        embeddings: Optional[Any] = None  # 可選：外部傳入的 embedding 模型（優先使用）
    ):
        """
        初始化向量檢索器（使用 Hugging Face embeddings）
        
        Args:
            documents: 文檔列表，每個文檔包含 "content" 和 "metadata"
            embedding_model: Hugging Face embedding 模型名稱（預設: "sentence-transformers/all-MiniLM-L6-v2"）
                            僅在 embeddings=None 時使用
            persist_directory: Chroma 資料庫持久化目錄
            hf_cache_dir: Hugging Face 模型緩存目錄（例如外接硬碟路徑）
                         如果為 None，則使用環境變數 HF_HOME 或默認位置 ~/.cache/huggingface/
                         僅在 embeddings=None 時使用
            device: 設備名稱 ('mps', 'cuda', 'cpu')，如果為 None 則自動檢測最佳設備
                   僅在 embeddings=None 時使用
            embeddings: 可選的外部 embedding 模型物件
                       如果提供，將優先使用此模型，忽略其他參數（embedding_model, hf_cache_dir, device）
                       這允許與 DocumentProcessor 共用同一個 embedding 模型實例
                       優點：
                       - 節省內存（只加載一次模型）
                       - 節省時間（避免重複初始化）
                       - 確保一致性（分塊和檢索使用相同的模型）
        """
        # 優先使用傳入的共用模型
        if embeddings is not None:
            self.embeddings = embeddings
            print("✓ 使用外部傳入的 embeddings 模型（與 DocumentProcessor 共用）")
        else:
            # 若無傳入，則執行原有的初始化邏輯
            print(f"使用 Hugging Face embedding 模型: {embedding_model}")
            
            # 設置 Hugging Face 緩存目錄
            if hf_cache_dir:
                # 如果指定了緩存目錄，設置環境變數
                os.environ['HF_HOME'] = hf_cache_dir
                os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
                print(f"模型將存儲在: {hf_cache_dir}")
            else:
                # 檢查是否已經設置了環境變數
                default_cache = os.path.expanduser("~/.cache/huggingface")
                current_cache = os.getenv('HF_HOME', default_cache)
                print(f"模型緩存位置: {current_cache}")
                print("提示: 可以通過設置 hf_cache_dir 參數或環境變數 HF_HOME 來指定外接硬碟路徑")
            
            # 自動檢測或使用指定的設備
            if device is None:
                device = get_device()
            
            device_name_map = {
                'mps': 'MPS (macOS GPU)',
                'cuda': 'CUDA (NVIDIA GPU)',
                'cpu': 'CPU'
            }
            print(f"使用設備: {device_name_map.get(device, device)}")
            print("首次使用時會下載模型，請稍候...")
            
            # 構建 model_kwargs，包含緩存目錄和設備
            model_kwargs = {'device': device}
            if hf_cache_dir:
                model_kwargs['cache_dir'] = hf_cache_dir
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}  # 正規化 embeddings 以提升效果
            )
        
        # 將文檔轉換為 LangChain Document 格式
        # 需要將 metadata 中的列表轉換為字串，因為 ChromaDB 不接受列表類型
        def sanitize_metadata(metadata: Dict) -> Dict:
            """將 metadata 中的列表轉換為字串，以符合 ChromaDB 的要求"""
            sanitized = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    # 將列表轉換為逗號分隔的字串
                    sanitized[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, (dict, set)):
                    # 將字典或集合轉換為字串
                    sanitized[key] = str(value)
                else:
                    # 其他類型（str, int, float, bool, None）直接保留
                    sanitized[key] = value
            return sanitized
        
        langchain_docs = [
            Document(
                page_content=doc["content"],
                metadata=sanitize_metadata(doc["metadata"])
            )
            for doc in documents
        ]
        
        # 創建向量資料庫
        self.vectorstore = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        # 創建 retriever
        self.retriever = self.vectorstore.as_retriever()
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        檢索相關文檔，並返回標準化的相似度分數（越高越好）。
        支援根據 metadata 進行過濾。
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件字典。
                            例如: {"arxiv_id": "1234.5678"} 只檢索特定論文的 chunks
                            或 {"title": "Machine Learning"} 只檢索特定標題的論文
                            支援多個條件，所有條件必須同時滿足（AND 邏輯）
                            注意：ChromaDB 的 where 條件支援精確匹配，不支援部分匹配
            
        Returns:
            相關文檔列表，每個包含 "content", "metadata", 和 "score"
            結果會根據 metadata_filter 進行過濾
        """
        # 構建過濾條件
        # 如果提供了 metadata_filter，先獲取更多結果，然後在 Python 中進行過濾
        # 這是因為 LangChain ChromaDB 的 similarity_search_with_score 方法
        # 對 filter 參數的支援可能因版本而異
        if metadata_filter:
            # 獲取更多結果以確保有足夠的候選進行過濾
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query, 
                k=top_k * 10  # 獲取更多結果
            )
            
            # 在 Python 中進行過濾
            filtered_results = []
            for doc, distance_score in results_with_scores:
                metadata = doc.metadata
                matches = True
                
                for key, value in metadata_filter.items():
                    doc_value = metadata.get(key)
                    
                    # 檢查是否匹配
                    if isinstance(value, dict):
                        # 支援運算符格式（例如 {"$eq": "value"}）
                        if "$eq" in value:
                            if doc_value != value["$eq"]:
                                matches = False
                                break
                        else:
                            # 其他運算符可以在此擴展
                            matches = False
                            break
                    elif isinstance(value, str) and isinstance(doc_value, str):
                        # 字串匹配：支援部分匹配（包含）
                        if value.lower() not in doc_value.lower():
                            matches = False
                            break
                    else:
                        # 其他類型使用精確匹配
                        if doc_value != value:
                            matches = False
                            break
                
                if matches:
                    filtered_results.append((doc, distance_score))
            
            # 只保留前 top_k 個結果
            results_with_scores = filtered_results[:top_k]
        else:
            # 沒有過濾條件，直接獲取結果
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query, 
                k=top_k
            )
        
        # 構建結果並轉換分數
        results = []
        for doc, distance_score in results_with_scores:
            # 因為 embedding 已正規化，L2 距離的平方為 2 - 2 * cos_sim
            # -> cos_sim = 1 - (distance^2 / 2)
            # 分數範圍在 [0, 1] 之間，越高越相似
            similarity_score = 1 - (distance_score**2 / 2)
            
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(similarity_score),
            })
        
        return results

