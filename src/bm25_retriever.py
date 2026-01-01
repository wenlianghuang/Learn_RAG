"""
BM25 檢索器模組
"""
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
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
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        檢索相關文檔
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            
        Returns:
            相關文檔列表，每個包含 "content", "metadata", "score"
        """
        # Tokenize 查詢
        tokenized_query = self._tokenize(query)
        
        # 計算 BM25 分數
        scores = self.bm25.get_scores(tokenized_query)
        
        # 獲取 top_k 個結果
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # 構建結果
        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(scores[idx]),
            })
        
        return results
    
    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        檢索相關文檔並返回分數
        
        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            
        Returns:
            (文檔, 分數) 元組列表
        """
        results = self.retrieve(query, top_k)
        return [(result, result["score"]) for result in results]

