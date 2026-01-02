"""
重排序模組：使用 Cross-Encoder 進行精準重排
"""
from typing import List, Dict, Optional, Tuple
from sentence_transformers import CrossEncoder
import time
import logging

# 嘗試導入 torch 來檢測可用的設備
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class Reranker:
    """重排序組件：使用 Cross-Encoder 進行精準重排"""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-base", 
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32,
        enable_cache: bool = True
    ):
        """
        初始化 Cross-Encoder 模型
        
        Args:
            model_name: Cross-Encoder 模型名稱
            device: 設備名稱 ('cuda', 'cpu', 'mps')
            max_length: 最大 token 長度（模型限制）
            batch_size: 批處理大小，用於優化內存使用
            enable_cache: 是否啟用模型緩存
        """
        try:
            # 自動檢測設備（如果未指定）
            if device is None:
                device = get_device()
            
            device_name_map = {
                'mps': 'MPS (macOS GPU)',
                'cuda': 'CUDA (NVIDIA GPU)',
                'cpu': 'CPU'
            }
            device_display = device_name_map.get(device, device)
            
            self.model = CrossEncoder(
                model_name, 
                device=device,
                max_length=max_length
            )
            self.max_length = max_length
            self.batch_size = batch_size
            self.model_name = model_name
            logger.info(f"✅ 重排模型 {model_name} 已載入 (device: {device_display})")
        except Exception as e:
            logger.error(f"❌ 模型載入失敗: {e}")
            raise
    
    def _truncate_text(self, text: str, max_chars: int = 2000) -> str:
        """
        截斷過長的文本（粗略估計，避免超過 token 限制）
        
        Args:
            text: 原始文本
            max_chars: 最大字符數（保守估計，約 500 tokens）
            
        Returns:
            截斷後的文本
        """
        if len(text) <= max_chars:
            return text
        # 截斷並添加省略號
        return text[:max_chars - 3] + "..."
    
    def _prepare_pairs(
        self, 
        query: str, 
        documents: List[Dict]
    ) -> List[Tuple[str, str]]:
        """
        準備 (query, document) 配對，處理文本長度
        
        Args:
            query: 查詢文本
            documents: 文檔列表
            
        Returns:
            (query, content) 配對列表
        """
        pairs = []
        truncated_indices = []  # 記錄哪些文檔被截斷了
        
        # 粗略估計：每個字符約 0.25 tokens，為 query 預留空間
        max_doc_chars = int((self.max_length * 0.7) - len(query))
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            original_length = len(content)
            
            # 如果內容過長，進行截斷
            if len(content) > max_doc_chars:
                content = self._truncate_text(content, max_doc_chars)
                truncated_indices.append(i)
            
            pairs.append([query, content])
        
        if truncated_indices:
            logger.warning(
                f"⚠️  有 {len(truncated_indices)} 個文檔因過長被截斷 "
                f"(最大長度: {max_doc_chars} 字符)"
            )
        
        return pairs
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 5,
        preserve_original_scores: bool = True
    ) -> List[Dict]:
        """
        執行精準重排
        
        Args:
            query: 查詢文本
            documents: 文檔列表，每個應包含 "content" 和可選的 "hybrid_score"
            top_k: 返回前 k 個結果
            preserve_original_scores: 是否保留原始分數（hybrid_score）
            
        Returns:
            重排後的文檔列表，按 rerank_score 降序排列
        """
        if not documents:
            logger.warning("⚠️  文檔列表為空，返回空結果")
            return []
        
        if not query or not query.strip():
            logger.warning("⚠️  查詢為空，返回原始文檔順序")
            return documents[:top_k]
        
        start_time = time.time()
        logger.info(f"🔄 開始重排 {len(documents)} 個文檔...")
        
        try:
            # 1. 準備配對
            pairs = self._prepare_pairs(query, documents)
            
            # 2. 批處理計算分數（優化內存使用）
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch_pairs)
                scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
            
            # 3. 更新文檔分數
            for i, doc in enumerate(documents):
                doc = doc.copy()  # 避免修改原始文檔
                doc["rerank_score"] = float(scores[i])
                
                # 保留原始分數供參考
                if preserve_original_scores:
                    if "hybrid_score" not in doc:
                        # 如果沒有 hybrid_score，嘗試使用其他分數
                        doc["original_score"] = doc.get("score", 0.0)
                
                documents[i] = doc
            
            # 4. 根據 rerank_score 重新排序
            reranked_docs = sorted(
                documents, 
                key=lambda x: x.get("rerank_score", float('-inf')), 
                reverse=True
            )
            
            # 5. 統計信息
            elapsed_time = time.time() - start_time
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0
            
            logger.info(
                f"✅ 重排完成 (耗時: {elapsed_time:.2f}s, "
                f"平均分數: {avg_score:.4f}, "
                f"範圍: [{min_score:.4f}, {max_score:.4f}])"
            )
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"❌ 重排過程出錯: {e}")
            # 降級策略：返回原始順序的前 top_k 個
            logger.warning("⚠️  使用降級策略：返回原始順序")
            return documents[:top_k]


class RAGPipeline:
    """協調管線：管理完整的 RAG 流程（召回 + 重排）"""
    
    def __init__(
        self, 
        hybrid_search, 
        reranker, 
        recall_k: int = 25,
        adaptive_recall: bool = True,
        min_recall_k: int = 10,
        max_recall_k: int = 50
    ):
        """
        初始化 RAG 管線
        
        Args:
            hybrid_search: HybridSearch 實例
            reranker: Reranker 實例
            recall_k: 第一階段召回的數量（預設值）
            adaptive_recall: 是否根據查詢動態調整 recall_k
            min_recall_k: 最小召回數量
            max_recall_k: 最大召回數量
        """
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.base_recall_k = recall_k
        self.adaptive_recall = adaptive_recall
        self.min_recall_k = min_recall_k
        self.max_recall_k = max_recall_k
        
        # 性能統計
        self.stats = {
            "total_queries": 0,
            "avg_recall_time": 0.0,
            "avg_rerank_time": 0.0,
            "avg_total_time": 0.0
        }
    
    def _calculate_adaptive_recall_k(self, query: str) -> int:
        """
        根據查詢複雜度動態計算 recall_k
        
        Args:
            query: 查詢文本
            
        Returns:
            調整後的 recall_k
        """
        if not self.adaptive_recall:
            return self.base_recall_k
        
        # 簡單啟發式：根據查詢長度和關鍵詞數量調整
        query_length = len(query.split())
        keyword_count = len(set(query.lower().split()))
        
        # 複雜查詢需要更多候選
        if query_length > 10 or keyword_count > 5:
            recall_k = min(self.base_recall_k * 2, self.max_recall_k)
        elif query_length < 3:
            recall_k = max(self.base_recall_k // 2, self.min_recall_k)
        else:
            recall_k = self.base_recall_k
        
        return recall_k
    
    def query(
        self, 
        text: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict] = None,
        enable_rerank: bool = True,
        return_stats: bool = False
    ) -> List[Dict]:
        """
        執行完整的搜尋流程
        
        Args:
            text: 查詢文本
            top_k: 最終返回的結果數量
            metadata_filter: 可選的 metadata 過濾條件
            enable_rerank: 是否啟用重排序（可選，用於性能測試）
            return_stats: 是否返回性能統計信息
            
        Returns:
            相關文檔列表，如果 return_stats=True，則返回 (results, stats) 元組
        """
        if not text or not text.strip():
            logger.warning("⚠️  查詢為空")
            return []
        
        total_start = time.time()
        self.stats["total_queries"] += 1
        
        # 動態計算 recall_k
        recall_k = self._calculate_adaptive_recall_k(text)
        logger.info(
            f"🔍 搜尋中: '{text[:50]}...' "
            f"(召回階段: {recall_k} 筆, 最終返回: {top_k} 筆)"
        )
        
        try:
            # 第一階段：混合搜尋（召回階段）
            recall_start = time.time()
            initial_results = self.hybrid_search.retrieve(
                query=text, 
                top_k=recall_k, 
                metadata_filter=metadata_filter
            )
            recall_time = time.time() - recall_start
            
            if not initial_results:
                logger.warning("⚠️  召回階段未找到任何結果")
                return []
            
            logger.info(
                f"✅ 召回階段完成: 找到 {len(initial_results)} 個候選 "
                f"(耗時: {recall_time:.2f}s)"
            )
            
            # 第二階段：重排序（精篩階段）
            if enable_rerank and len(initial_results) > top_k:
                rerank_start = time.time()
                final_results = self.reranker.rerank(
                    query=text, 
                    documents=initial_results, 
                    top_k=top_k
                )
                rerank_time = time.time() - rerank_start
                
                logger.info(
                    f"✅ 重排階段完成: 從 {len(initial_results)} 個候選中選出 "
                    f"{len(final_results)} 個結果 (耗時: {rerank_time:.2f}s)"
                )
            else:
                # 跳過重排序（用於性能測試或候選數較少時）
                final_results = initial_results[:top_k]
                rerank_time = 0.0
                logger.info("⏭️  跳過重排序階段（候選數不足或已禁用）")
            
            # 更新統計信息
            total_time = time.time() - total_start
            self._update_stats(recall_time, rerank_time, total_time)
            
            # 添加性能信息到結果（可選）
            if return_stats:
                stats = {
                    "recall_time": recall_time,
                    "rerank_time": rerank_time,
                    "total_time": total_time,
                    "recall_k": recall_k,
                    "candidates_found": len(initial_results),
                    "final_results": len(final_results)
                }
                return final_results, stats
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 查詢過程出錯: {e}")
            # 降級策略：嘗試只使用召回階段
            try:
                logger.warning("⚠️  嘗試降級策略：僅使用召回結果")
                return self.hybrid_search.retrieve(text, top_k=top_k, metadata_filter=metadata_filter)
            except Exception as e2:
                logger.error(f"❌ 降級策略也失敗: {e2}")
                return []
    
    def _update_stats(self, recall_time: float, rerank_time: float, total_time: float):
        """更新性能統計信息"""
        n = self.stats["total_queries"]
        self.stats["avg_recall_time"] = (
            (self.stats["avg_recall_time"] * (n - 1) + recall_time) / n
        )
        self.stats["avg_rerank_time"] = (
            (self.stats["avg_rerank_time"] * (n - 1) + rerank_time) / n
        )
        self.stats["avg_total_time"] = (
            (self.stats["avg_total_time"] * (n - 1) + total_time) / n
        )
    
    def get_stats(self) -> Dict:
        """獲取性能統計信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置統計信息"""
        self.stats = {
            "total_queries": 0,
            "avg_recall_time": 0.0,
            "avg_rerank_time": 0.0,
            "avg_total_time": 0.0
        }
    
    def format_results_for_llm(
        self,
        results: List[Dict],
        format_style: str = "detailed"
    ) -> str:
        """
        格式化檢索結果供 LLM 使用（需要導入 PromptFormatter）
        
        Args:
            results: 檢索結果列表
            format_style: 格式風格 ("detailed", "simple", "minimal")
            
        Returns:
            格式化後的上下文字符串
        """
        try:
            from ..prompt_formatter import PromptFormatter
            formatter = PromptFormatter(format_style=format_style)
            return formatter.format_context(results)
        except ImportError:
            # 如果無法導入，使用簡單格式
            formatted_parts = []
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                content = result.get("content", "")
                arxiv_id = metadata.get('arxiv_id', 'N/A')
                title = metadata.get('title', 'N/A')
                formatted_parts.append(
                    f"[來源 {i}: {title} (arXiv:{arxiv_id})]\n{content}\n"
                )
            return "\n" + "="*60 + "\n".join(formatted_parts)

