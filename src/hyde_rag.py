"""
HyDE (Hypothetical Document Embeddings) RAG：使用假設性文檔改善檢索
"""
from typing import List, Dict, Optional
from .retrievers.reranker import RAGPipeline
from .retrievers.vector_retriever import VectorRetriever
from .prompt_formatter import PromptFormatter
from .llm_integration import OllamaLLM
import time
import logging

logger = logging.getLogger(__name__)


class HyDERAG:
    """使用 HyDE (Hypothetical Document Embeddings) 的 RAG 系統"""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_retriever: VectorRetriever,
        llm: OllamaLLM,
        hypothetical_length: int = 200,
        temperature: float = 0.7
    ):
        """
        初始化 HyDE RAG
        
        Args:
            rag_pipeline: RAG 管線實例（用於最終答案生成）
            vector_retriever: 向量檢索器（用於基於假設性文檔的檢索）
            llm: LLM 實例（用於生成假設性文檔）
            hypothetical_length: 假設性文檔的目標長度（字符數）
            temperature: 生成假設性文檔時的溫度參數（建議 0.7，以獲得更多專業術語）
        """
        self.rag_pipeline = rag_pipeline
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.hypothetical_length = hypothetical_length
        self.temperature = temperature
    
    def _generate_hypothetical_document(self, question: str) -> str:
        """
        生成假設性文檔（Hypothetical Document）
        
        Args:
            question: 用戶問題
            
        Returns:
            假設性文檔文本
        """
        # 檢測語言
        is_chinese = PromptFormatter.detect_language(question) == "zh"
        
        if is_chinese:
            prompt = f"""請針對以下問題，寫出一段約 {self.hypothetical_length} 字的專業技術文件內容。
這段內容應包含該領域常見的專業術語與原理說明，以便用於後續的語義檢索。
請使用專業的術語和概念，即使你對某些細節不確定，也要包含相關的專業詞彙。

問題: {question}

專業技術內容："""
        else:
            prompt = f"""Please write a professional technical document of approximately {self.hypothetical_length} words in response to the following question.
This content should include common professional terminology and principle explanations in this field, to be used for subsequent semantic retrieval.
Please use professional terms and concepts, and include relevant professional vocabulary even if you are uncertain about some details.

Question: {question}

Professional technical content:"""
        
        try:
            hypothetical_doc = self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,  # 較高的溫度以獲得更多專業術語
                max_tokens=500
            )
            
            # 清理輸出
            hypothetical_doc = hypothetical_doc.strip()
            
            if not hypothetical_doc:
                logger.warning("⚠️  生成的假設性文檔為空，使用原始問題")
                return question
            
            logger.info(f"✅ 生成假設性文檔（長度: {len(hypothetical_doc)} 字符）")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"⚠️  生成假設性文檔時出錯: {e}")
            # 回退到使用原始問題
            return question
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        return_hypothetical: bool = False
    ) -> Dict:
        """
        執行 HyDE 檢索（不生成答案）
        
        Args:
            question: 原始問題
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件
            return_hypothetical: 是否在結果中包含假設性文檔
            
        Returns:
            包含檢索結果和統計信息的字典
        """
        start_time = time.time()
        
        # 第一步：生成假設性文檔
        logger.info(f"🔍 生成假設性文檔: '{question}'")
        hypothetical_doc = self._generate_hypothetical_document(question)
        
        # 第二步：使用假設性文檔進行檢索
        logger.info(f"📚 使用假設性文檔進行檢索...")
        results = self.vector_retriever.retrieve(
            query=hypothetical_doc,  # 使用假設性文檔而不是原始問題
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 找到 {len(results)} 個結果（耗時: {elapsed_time:.2f}s）")
        
        result = {
            "results": results,
            "total_docs_found": len(results),
            "hypothetical_document": hypothetical_doc if return_hypothetical else None,
            "elapsed_time": elapsed_time
        }
        
        return result
    
    def generate_answer(
        self,
        question: str,
        formatter: PromptFormatter,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        document_type: str = "general",
        return_hypothetical: bool = False
    ) -> Dict:
        """
        完整的 HyDE RAG 流程：生成假設性文檔 -> 檢索 -> 生成答案
        
        Args:
            question: 原始問題
            formatter: Prompt 格式化器
            top_k: 用於生成答案的文檔數量
            metadata_filter: 可選的 metadata 過濾條件
            document_type: 文檔類型 ("paper", "cv", "general")
            return_hypothetical: 是否在結果中包含假設性文檔
            
        Returns:
            包含檢索結果、生成的答案和統計信息的字典
        """
        start_time = time.time()
        
        # 第一步：生成假設性文檔
        logger.info(f"🔍 生成假設性文檔: '{question}'")
        hypothetical_start = time.time()
        hypothetical_doc = self._generate_hypothetical_document(question)
        hypothetical_time = time.time() - hypothetical_start
        
        # 第二步：使用假設性文檔進行檢索
        logger.info(f"📚 使用假設性文檔進行檢索...")
        retrieval_start = time.time()
        results = self.vector_retriever.retrieve(
            query=hypothetical_doc,  # 使用假設性文檔而不是原始問題
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        retrieval_time = time.time() - retrieval_start
        
        if not results:
            return {
                "results": [],
                "total_docs_found": 0,
                "hypothetical_document": hypothetical_doc if return_hypothetical else None,
                "elapsed_time": retrieval_time + hypothetical_time,
                "answer": "抱歉，未找到相關文檔來回答此問題。",
                "formatted_context": None,
                "answer_time": 0.0,
                "total_time": retrieval_time + hypothetical_time
            }
        
        # 第三步：格式化上下文
        formatted_context = formatter.format_context(
            results,
            document_type=document_type
        )
        
        # 第四步：創建 prompt（使用原始問題，而不是假設性文檔）
        prompt = formatter.create_prompt(
            question,  # 使用原始問題生成答案
            formatted_context,
            document_type=document_type
        )
        
        # 第五步：生成回答
        logger.info("🤖 生成回答中...")
        answer_start = time.time()
        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2048
            )
            answer_time = time.time() - answer_start
            logger.info(f"✅ 回答生成完成（耗時: {answer_time:.2f}s）")
        except Exception as e:
            logger.error(f"❌ 生成回答時出錯: {e}")
            answer = f"生成回答時出錯: {e}"
            answer_time = time.time() - answer_start
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "total_docs_found": len(results),
            "hypothetical_document": hypothetical_doc if return_hypothetical else None,
            "elapsed_time": retrieval_time + hypothetical_time,
            "hypothetical_time": hypothetical_time,
            "retrieval_time": retrieval_time,
            "answer": answer,
            "formatted_context": formatted_context,
            "answer_time": answer_time,
            "total_time": total_time
        }

