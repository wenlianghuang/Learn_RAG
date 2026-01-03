"""
BM25 檢索器模組
"""
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import re
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """使用 BM25 演算法進行關鍵字檢索"""
    
    def __init__(self, documents: List[Dict]):
        """
        初始化 BM25 檢索器
        
        Args:
            documents: 文檔列表，每個文檔包含 "content" 和 "metadata"
        """
        self.documents = documents
        self.texts = [doc["content"] for doc in documents]
        
        # 對文字進行 tokenization（簡單的分詞）
        tokenized_texts = [self._tokenize(text) for text in self.texts]
        
        # 初始化 BM25
        self.bm25 = BM25Okapi(tokenized_texts)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        將文字轉換為 tokens（簡單的實作）
        
        Args:
            text: 輸入文字
            
        Returns:
            token 列表
        """
        # 轉為小寫並分割
        text = text.lower()
        # 使用正則表達式分割（保留字母和數字）
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        檢索相關文檔，支援根據 metadata 進行過濾。

        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件字典。
                            例如: {"arxiv_id": "1234.5678"} 只檢索特定論文的 chunks
                            或 {"title": "Machine Learning"} 只檢索特定標題的論文
                            支援多個條件，所有條件必須同時滿足（AND 邏輯）
                            注意：BM25 的過濾是在檢索後進行的，所以可能會返回少於 top_k 的結果

        Returns:
            相關文檔列表，每個包含 "content", "metadata", "score"
            結果會根據 metadata_filter 進行過濾
        """
        # Tokenize 查詢
        tokenized_query = self._tokenize(query)
        
        # 計算 BM25 分數
        scores = self.bm25.get_scores(tokenized_query)
        
        # 獲取所有結果並排序（先獲取更多結果以應對過濾後可能減少的情況）
        # 如果沒有過濾條件，只需要 top_k 個；如果有過濾條件，需要更多候選結果
        candidate_k = top_k * 3 if metadata_filter else top_k
        
        # 獲取候選結果索引（按分數降序排列）
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:candidate_k]
        
        # 構建候選結果
        candidate_results = []
        for idx in sorted_indices:
            candidate_results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(scores[idx]),
            })
        
        # 如果提供了 metadata_filter，則進行過濾
        if metadata_filter:
            filtered_results = []
            for result in candidate_results:
                # 檢查該結果的 metadata 是否滿足所有過濾條件
                metadata = result.get("metadata", {})
                matches_all = True
                
                for filter_key, filter_value in metadata_filter.items():
                    # 獲取文檔中對應的 metadata 值
                    doc_value = metadata.get(filter_key)
                    
                    # 檢查是否匹配
                    # 支援精確匹配和部分匹配（如果 filter_value 是字串且 doc_value 也是字串）
                    if isinstance(filter_value, str) and isinstance(doc_value, str):
                        # 字串匹配：支援精確匹配或包含匹配
                        if filter_value.lower() not in doc_value.lower():
                            matches_all = False
                            break
                    else:
                        # 其他類型（數字、布林值等）使用精確匹配
                        if doc_value != filter_value:
                            matches_all = False
                            break
                
                # 如果所有條件都滿足，則加入結果
                if matches_all:
                    filtered_results.append(result)
            
            # 返回過濾後的結果（最多 top_k 個）
            return filtered_results[:top_k]
        else:
            # 沒有過濾條件，直接返回候選結果
            return candidate_results

