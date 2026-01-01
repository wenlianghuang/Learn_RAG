"""
Hybrid Search 模組：結合 BM25 和向量檢索
"""
from typing import List, Dict, Tuple
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever
import numpy as np


class HybridSearch:
    """結合 BM25 和向量檢索的混合搜尋"""
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        """
        初始化 Hybrid Search
        
        Args:
            bm25_retriever: BM25 檢索器
            vector_retriever: 向量檢索器
            bm25_weight: BM25 分數的權重
            vector_weight: 向量分數的權重
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # 確保權重總和為 1
        total_weight = bm25_weight + vector_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.bm25_weight = bm25_weight / total_weight
            self.vector_weight = vector_weight / total_weight
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        正規化分數到 [0, 1] 區間
        
        Args:
            scores: 原始分數列表
            
        Returns:
            正規化後的分數列表
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _invert_scores(self, scores: List[float]) -> List[float]:
        """
        反轉分數（距離越小越好，分數越大越好）
        對於向量檢索，距離越小表示越相似，所以需要反轉
        
        Args:
            scores: 距離分數列表（越小越好）
            
        Returns:
            反轉後的分數列表（越大越好）
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        # 使用負數或倒數來反轉
        # 這裡使用 max - score 的方式
        max_score = scores_array.max()
        if max_score == 0:
            return [1.0] * len(scores)
        
        inverted = max_score - scores_array + 1e-6  # 加小值避免為 0
        return self._normalize_scores(inverted.tolist())
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        執行混合搜尋
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            
        Returns:
            相關文檔列表，每個包含 "content", "metadata", "hybrid_score"
        """
        # 從 BM25 檢索
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        bm25_scores = {i: result["score"] for i, result in enumerate(bm25_results)}
        
        # 從向量檢索
        vector_results_with_scores = self.vector_retriever.similarity_search_with_score(
            query, top_k=top_k * 2
        )
        vector_results = [result[0] for result in vector_results_with_scores]
        vector_scores = [result[1] for result in vector_results_with_scores]
        
        # 正規化分數
        bm25_normalized = self._normalize_scores(list(bm25_scores.values()))
        # 向量分數是距離，需要反轉
        vector_normalized = self._invert_scores(vector_scores)
        
        # 創建文檔 ID 到索引的映射
        doc_to_hybrid_score = {}
        
        # 處理 BM25 結果
        for i, (result, score) in enumerate(zip(bm25_results, bm25_normalized)):
            doc_id = self._get_doc_id(result)
            if doc_id not in doc_to_hybrid_score:
                doc_to_hybrid_score[doc_id] = {
                    "doc": result,
                    "bm25_score": score,
                    "vector_score": 0.0,
                }
            else:
                doc_to_hybrid_score[doc_id]["bm25_score"] = score
        
        # 處理向量結果
        for i, (result, score) in enumerate(zip(vector_results, vector_normalized)):
            doc_id = self._get_doc_id(result)
            if doc_id not in doc_to_hybrid_score:
                doc_to_hybrid_score[doc_id] = {
                    "doc": result,
                    "bm25_score": 0.0,
                    "vector_score": score,
                }
            else:
                doc_to_hybrid_score[doc_id]["vector_score"] = score
        
        # 計算混合分數
        hybrid_results = []
        for doc_id, data in doc_to_hybrid_score.items():
            hybrid_score = (
                self.bm25_weight * data["bm25_score"] +
                self.vector_weight * data["vector_score"]
            )
            result = data["doc"].copy()
            result["hybrid_score"] = hybrid_score
            result["bm25_score"] = data["bm25_score"]
            result["vector_score"] = data["vector_score"]
            hybrid_results.append(result)
        
        # 按混合分數排序並返回 top_k
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]
    
    def _get_doc_id(self, doc: Dict) -> str:
        """
        從文檔中獲取唯一 ID
        
        Args:
            doc: 文檔字典
            
        Returns:
            文檔唯一 ID
        """
        metadata = doc.get("metadata", {})
        # 使用 arxiv_id 和 chunk_index 作為唯一 ID
        arxiv_id = metadata.get("arxiv_id", "unknown")
        chunk_index = metadata.get("chunk_index", 0)
        return f"{arxiv_id}_{chunk_index}"

