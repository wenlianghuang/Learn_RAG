"""
Hybrid Search 模組：結合 BM25 和向量檢索
支援兩種融合方法：加權求和（Weighted Sum）和倒數排名融合（RRF）
"""
from typing import List, Dict, Optional, Literal
from .base import BaseRetriever
import numpy as np


class HybridSearch(BaseRetriever):
    """結合稀疏和密集檢索的混合搜尋"""
    
    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        sparse_weight: float = 0.4,
        dense_weight: float = 0.6,
        fusion_method: Literal["weighted_sum", "rrf"] = "rrf",
        rrf_k: int = 60,
    ):
        """
        初始化 Hybrid Search
        
        Args:
            sparse_retriever: 稀疏檢索器 (例如 BM25)
            dense_retriever: 密集檢索器 (例如向量檢索)
            sparse_weight: 稀疏檢索分數的權重（僅用於 weighted_sum 方法）
            dense_weight: 密集檢索分數的權重（僅用於 weighted_sum 方法）
            fusion_method: 融合方法，可選 "weighted_sum" 或 "rrf"
                          - "weighted_sum": 加權求和，需要正規化分數並設置權重
                          - "rrf": 倒數排名融合（Reciprocal Rank Fusion），
                                  不需要分數正規化，對不同分數分佈更魯棒
            rrf_k: RRF 方法中的常數 k，通常設為 60（僅用於 rrf 方法）
                   較大的 k 值會讓排名較低的文檔獲得更多權重
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        
        # 僅在 weighted_sum 方法中使用權重
        if fusion_method == "weighted_sum":
            self.sparse_weight = sparse_weight
            self.dense_weight = dense_weight
            
            # 確保權重總和為 1
            total_weight = sparse_weight + dense_weight
            if abs(total_weight - 1.0) > 1e-6:
                self.sparse_weight = sparse_weight / total_weight
                self.dense_weight = dense_weight / total_weight
            
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        將分數正規化到 [0, 1] 區間。
        僅用於 weighted_sum 方法。
        
        Args:
            results: 檢索結果列表，每個字典包含 'score'
            
        Returns:
            帶有正規化分數的結果列表
        """
        scores = [res.get("score", 0.0) for res in results]
        if not scores:
            return results
            
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            # 如果所有分數都相同，將它們設置為 1.0
            normalized_scores = [1.0] * len(scores)
        else:
            normalized_scores = ((scores_array - min_score) / (max_score - min_score)).tolist()
            
        for i, res in enumerate(results):
            res["score"] = normalized_scores[i]
            
        return results
    
    def _get_doc_id(self, doc: Dict) -> str:
        """
        從文檔中提取唯一標識符
        
        Args:
            doc: 文檔字典
            
        Returns:
            文檔的唯一 ID
        """
        metadata = doc.get("metadata", {})
        return f"{metadata.get('arxiv_id', 'unknown')}_{metadata.get('chunk_index', 0)}"
    
    def _apply_rrf(
        self, 
        sparse_results: List[Dict], 
        dense_results: List[Dict]
    ) -> List[Dict]:
        """
        應用倒數排名融合（Reciprocal Rank Fusion, RRF）方法
        
        RRF 公式：RRF(d) = Σ(1 / (k + rank_i(d)))
        其中：
        - d 是文檔
        - rank_i(d) 是文檔在第 i 個檢索結果中的排名（從 1 開始）
        - k 是常數（預設為 60）
        
        RRF 的優點：
        1. 不需要分數正規化，對不同分數分佈的檢索器更魯棒
        2. 只依賴排名位置，不依賴分數值
        3. 自動處理分數分佈差異的問題
        
        Args:
            sparse_results: 稀疏檢索結果列表
            dense_results: 密集檢索結果列表
            
        Returns:
            融合後的結果列表，按 RRF 分數排序
        """
        # 建立文檔 ID 到 RRF 分數的映射
        doc_to_rrf_score = {}
        
        # 處理稀疏檢索結果（BM25）
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = self._get_doc_id(result)
            if doc_id not in doc_to_rrf_score:
                doc_to_rrf_score[doc_id] = {
                    "doc": result,
                    "rrf_score": 0.0,
                    "sparse_rank": None,
                    "dense_rank": None
                }
            # 計算 RRF 貢獻：1 / (k + rank)
            doc_to_rrf_score[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)
            doc_to_rrf_score[doc_id]["sparse_rank"] = rank
        
        # 處理密集檢索結果（向量）
        for rank, result in enumerate(dense_results, start=1):
            doc_id = self._get_doc_id(result)
            if doc_id not in doc_to_rrf_score:
                doc_to_rrf_score[doc_id] = {
                    "doc": result,
                    "rrf_score": 0.0,
                    "sparse_rank": None,
                    "dense_rank": None
                }
            # 計算 RRF 貢獻：1 / (k + rank)
            doc_to_rrf_score[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)
            doc_to_rrf_score[doc_id]["dense_rank"] = rank
        
        # 構建結果列表
        rrf_results = []
        for doc_id, data in doc_to_rrf_score.items():
            result = data["doc"].copy()
            result["hybrid_score"] = data["rrf_score"]
            result["rrf_score"] = data["rrf_score"]
            result["sparse_rank"] = data["sparse_rank"]
            result["dense_rank"] = data["dense_rank"]
            
            # 從原始結果中獲取分數以供參考
            if data["sparse_rank"] is not None:
                # 從稀疏檢索結果中獲取原始分數
                for sparse_res in sparse_results:
                    if self._get_doc_id(sparse_res) == doc_id:
                        result["sparse_score"] = sparse_res.get("score", 0.0)
                        break
            else:
                result["sparse_score"] = None
                
            if data["dense_rank"] is not None:
                # 從密集檢索結果中獲取原始分數
                for dense_res in dense_results:
                    if self._get_doc_id(dense_res) == doc_id:
                        result["dense_score"] = dense_res.get("score", 0.0)
                        break
            else:
                result["dense_score"] = None
            
            rrf_results.append(result)
        
        # 按 RRF 分數從高到低排序
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        return rrf_results
    
    def _apply_weighted_sum(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict]
    ) -> List[Dict]:
        """
        應用加權求和（Weighted Sum）方法
        
        此方法需要：
        1. 正規化兩組分數到相同範圍
        2. 根據權重進行加權求和
        
        Args:
            sparse_results: 稀疏檢索結果列表
            dense_results: 密集檢索結果列表
            
        Returns:
            融合後的結果列表，按混合分數排序
        """
        # 正規化兩組分數
        normalized_sparse = self._normalize_scores(sparse_results)
        normalized_dense = self._normalize_scores(dense_results)
        
        # 結合分數
        doc_to_scores = {}
        
        # 處理稀疏檢索結果
        for res in normalized_sparse:
            doc_id = self._get_doc_id(res)
            if doc_id not in doc_to_scores:
                doc_to_scores[doc_id] = {"doc": res, "sparse": 0.0, "dense": 0.0}
            doc_to_scores[doc_id]["sparse"] = res["score"]
        
        # 處理密集檢索結果
        for res in normalized_dense:
            doc_id = self._get_doc_id(res)
            if doc_id not in doc_to_scores:
                doc_to_scores[doc_id] = {"doc": res, "sparse": 0.0, "dense": 0.0}
            doc_to_scores[doc_id]["dense"] = res["score"]
        
        # 計算混合分數並排序
        hybrid_results = []
        for doc_id, scores in doc_to_scores.items():
            hybrid_score = (
                self.sparse_weight * scores["sparse"] +
                self.dense_weight * scores["dense"]
            )
            
            result = scores["doc"].copy()
            result["hybrid_score"] = hybrid_score
            result["sparse_score"] = scores["sparse"]
            result["dense_score"] = scores["dense"]
            hybrid_results.append(result)
        
        # 按混合分數從高到低排序
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return hybrid_results

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        執行混合搜尋，支援根據 metadata 進行過濾
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件字典。
                            例如: {"arxiv_id": "1234.5678"} 只檢索特定論文的 chunks
                            或 {"title": "Machine Learning"} 只檢索特定標題的論文
                            支援多個條件，所有條件必須同時滿足（AND 邏輯）
                            此過濾條件會傳遞給底層的稀疏和密集檢索器
            
        Returns:
            相關文檔列表，每個包含 "content", "metadata", "hybrid_score"
            結果會根據 metadata_filter 進行過濾
            
            根據 fusion_method 的不同，返回的結果會包含不同的分數欄位：
            - RRF 方法：包含 "rrf_score", "sparse_rank", "dense_rank"
            - Weighted Sum 方法：包含 "sparse_score", "dense_score"
        """
        # 1. 從兩個檢索器獲取結果（請求更多結果以確保覆蓋率）
        # 將 metadata_filter 傳遞給底層檢索器
        sparse_results = self.sparse_retriever.retrieve(
            query, 
            top_k=top_k * 2,
            metadata_filter=metadata_filter
        )
        dense_results = self.dense_retriever.retrieve(
            query, 
            top_k=top_k * 2,
            metadata_filter=metadata_filter
        )
        
        # 2. 根據選擇的融合方法進行結果融合
        if self.fusion_method == "rrf":
            # 使用 RRF（倒數排名融合）方法
            # RRF 不需要分數正規化，直接基於排名進行融合
            hybrid_results = self._apply_rrf(sparse_results, dense_results)
        else:
            # 使用加權求和方法
            # 需要先正規化分數，然後根據權重進行加權求和
            hybrid_results = self._apply_weighted_sum(sparse_results, dense_results)
        
        # 3. 返回前 top_k 個結果
        return hybrid_results[:top_k]

