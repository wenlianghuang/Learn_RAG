"""
向量檢索器模組：使用 embedding 和向量資料庫進行語義檢索
"""
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

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


class VectorRetriever:
    """使用向量檢索進行語義搜尋"""
    
    def __init__(
        self,
        documents: List[Dict],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = "./chroma_db",
        hf_cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化向量檢索器（使用 Hugging Face embeddings）
        
        Args:
            documents: 文檔列表，每個文檔包含 "content" 和 "metadata"
            embedding_model: Hugging Face embedding 模型名稱（預設: "sentence-transformers/all-MiniLM-L6-v2"）
            persist_directory: Chroma 資料庫持久化目錄
            hf_cache_dir: Hugging Face 模型緩存目錄（例如外接硬碟路徑）
                         如果為 None，則使用環境變數 HF_HOME 或默認位置 ~/.cache/huggingface/
            device: 設備名稱 ('mps', 'cuda', 'cpu')，如果為 None 則自動檢測最佳設備
        """
        # 使用 Hugging Face embeddings（本地運行，完全免費）
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
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        檢索相關文檔
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            
        Returns:
            相關文檔列表，每個包含 "content", "metadata", "score"
        """
        # 使用 similarity_search_with_score 來獲得分數
        # 注意：這裡的分數是距離分數（越小越相似），與 BM25 的分數（越大越相關）不同
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # 構建結果
        results = []
        for doc, score in results_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),  # 轉換為 float，這是距離分數（越小越相似）
            })
        
        return results
    
    def similarity_search_with_score(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        檢索相關文檔並返回相似度分數
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            
        Returns:
            (文檔, 分數) 元組列表
        """
        # 使用 similarity_search_with_score 獲取分數
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append((
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                },
                float(score)
            ))
        
        return formatted_results

