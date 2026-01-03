"""
檢索器模組的抽象基類
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseRetriever(ABC):
    """檢索器的抽象基類"""

    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        檢索相關文檔並返回帶有分數的結果。

        Args:
            query: 查詢文字
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件字典。
                            例如: {"arxiv_id": "1234.5678"} 或 {"title": "Machine Learning"}
                            支援多個條件，所有條件必須同時滿足（AND 邏輯）

        Returns:
            相關文檔列表，每個文檔字典都應包含 "score" 鍵，
            且分數越高代表越相關。返回的結果會根據 metadata_filter 進行過濾。
        """
        pass
