"""
Prompt 格式化模組：將檢索結果格式化為 LLM 可讀的上下文
"""
from typing import List, Dict, Optional
import re


class PromptFormatter:
    """格式化檢索結果供 LLM 使用"""
    
    def __init__(
        self,
        include_metadata: bool = True,
        format_style: str = "detailed",
        max_context_length: Optional[int] = None,
        auto_detect_language: bool = True
    ):
        """
        初始化 Prompt 格式化器
        
        Args:
            include_metadata: 是否包含來源資訊
            format_style: 格式風格 ("detailed", "simple", "minimal")
            max_context_length: 最大上下文長度（字符數），None 表示不限制
            auto_detect_language: 是否自動檢測語言並相應調整回答語言
        """
        self.include_metadata = include_metadata
        self.format_style = format_style
        self.max_context_length = max_context_length
        self.auto_detect_language = auto_detect_language
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        檢測文本的主要語言
        
        Args:
            text: 輸入文本
            
        Returns:
            "zh" 表示中文，"en" 表示英文
        """
        # 檢查是否包含中文字符（CJK 統一表意文字範圍）
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        chinese_chars = len(chinese_pattern.findall(text))
        
        # 計算中文字符比例
        total_chars = len([c for c in text if c.isalnum() or c.isspace()])
        
        if total_chars == 0:
            return "en"  # 預設英文
        
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        # 如果中文字符比例超過 20%，認為是中文
        if chinese_ratio > 0.2:
            return "zh"
        else:
            return "en"
    
    def get_system_prompt(self, language: str = "zh", document_type: str = "general") -> str:
        """
        根據語言和文檔類型獲取系統提示詞
        
        Args:
            language: 語言代碼 ("zh" 或 "en")
            document_type: 文檔類型 ("paper", "cv", "general")
                          "paper": 學術論文
                          "cv": 履歷/履歷
                          "general": 通用文檔（預設）
            
        Returns:
            系統提示詞字符串
        """
        if language == "zh":
            if document_type == "paper":
                return (
                    "你是一個專業的 AI 研究助手，專門回答關於機器學習、"
                    "深度學習和自然語言處理的問題。\n\n"
                    "請基於以下提供的學術論文片段來回答用戶的問題。"
                    "每個片段都標註了來源論文的資訊。\n\n"
                    "回答要求：\n"
                    "1. 基於提供的上下文回答問題\n"
                    "2. 如果上下文不足以回答，請明確說明\n"
                    "3. 在回答中引用具體的論文來源（使用 arXiv ID）\n"
                    "4. 如果不同論文有不同觀點，請分別說明\n"
                    "5. 保持回答簡潔、準確、專業\n"
                    "6. **重要：請使用與用戶問題相同的語言回答**\n"
                )
            elif document_type == "cv":
                return (
                    "你是一個專業的 AI 助手，專門幫助分析和介紹簡歷（CV）內容。\n\n"
                    "請基於以下提供的文檔片段來回答用戶的問題。"
                    "這些片段來自一份簡歷或履歷表。\n\n"
                    "回答要求：\n"
                    "1. 基於提供的上下文回答問題\n"
                    "2. 如果上下文不足以回答，請明確說明\n"
                    "3. 在回答中引用具體的文檔內容\n"
                    "4. 保持回答簡潔、準確、專業\n"
                    "5. **重要：請使用與用戶問題相同的語言回答**\n"
                    "6. **請理解：這些片段就是簡歷的內容，請直接基於這些內容回答問題**\n"
                )
            else:  # general
                return (
                    "你是一個專業的 AI 助手。\n\n"
                    "請基於以下提供的文檔片段來回答用戶的問題。"
                    "每個片段都標註了來源資訊。\n\n"
                    "回答要求：\n"
                    "1. 基於提供的上下文回答問題\n"
                    "2. 如果上下文不足以回答，請明確說明\n"
                    "3. 在回答中引用具體的文檔內容\n"
                    "4. 保持回答簡潔、準確、專業\n"
                    "5. **重要：請使用與用戶問題相同的語言回答**\n"
                )
        else:  # English
            if document_type == "paper":
                return (
                    "You are a professional AI research assistant specializing in "
                    "machine learning, deep learning, and natural language processing.\n\n"
                    "Please answer the user's question based on the provided academic paper excerpts. "
                    "Each excerpt is labeled with source paper information.\n\n"
                    "Answer requirements:\n"
                    "1. Answer the question based on the provided context\n"
                    "2. If the context is insufficient, clearly state so\n"
                    "3. Cite specific paper sources in your answer (using arXiv ID)\n"
                    "4. If different papers have different viewpoints, explain them separately\n"
                    "5. Keep answers concise, accurate, and professional\n"
                    "6. **Important: Please answer in the same language as the user's question**\n"
                )
            elif document_type == "cv":
                return (
                    "You are a professional AI assistant specializing in analyzing and introducing CV (Curriculum Vitae) content.\n\n"
                    "Please answer the user's question based on the provided document excerpts. "
                    "These excerpts are from a CV or resume.\n\n"
                    "Answer requirements:\n"
                    "1. Answer the question based on the provided context\n"
                    "2. If the context is insufficient, clearly state so\n"
                    "3. Cite specific document content in your answer\n"
                    "4. Keep answers concise, accurate, and professional\n"
                    "5. **Important: Please answer in the same language as the user's question**\n"
                    "6. **Please understand: These excerpts ARE the CV content. Please answer directly based on this content.**\n"
                )
            else:  # general
                return (
                    "You are a professional AI assistant.\n\n"
                    "Please answer the user's question based on the provided document excerpts. "
                    "Each excerpt is labeled with source information.\n\n"
                    "Answer requirements:\n"
                    "1. Answer the question based on the provided context\n"
                    "2. If the context is insufficient, clearly state so\n"
                    "3. Cite specific document content in your answer\n"
                    "4. Keep answers concise, accurate, and professional\n"
                    "5. **Important: Please answer in the same language as the user's question**\n"
                )
    
    def format_context(
        self, 
        results: List[Dict], 
        include_metadata: Optional[bool] = None,
        format_style: Optional[str] = None,
        document_type: str = "general"
    ) -> str:
        """
        格式化檢索結果為 LLM 可讀的上下文
        
        Args:
            results: 檢索結果列表
            include_metadata: 是否包含來源資訊（覆蓋初始化參數）
            format_style: 格式風格（覆蓋初始化參數）
            document_type: 文檔類型 ("paper", "cv", "general")，用於調整格式
            
        Returns:
            格式化後的上下文字符串
        """
        if include_metadata is None:
            include_metadata = self.include_metadata
        if format_style is None:
            format_style = self.format_style
        
        if not results:
            # 根據格式風格選擇語言
            if format_style == "detailed" or format_style == "simple":
                return "（未找到相關文檔片段）"
            else:
                return "(No relevant excerpts found)"
        
        formatted_parts = []
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            if not include_metadata:
                # 不包含來源資訊，直接使用內容
                formatted_parts.append(f"{content}\n")
            elif format_style == "detailed":
                # 詳細格式：根據文檔類型調整顯示資訊
                if document_type == "cv":
                    # CV 格式：顯示檔案名和路徑
                    source_info = (
                        f"[來源 {i}]\n"
                        f"檔案標題: {metadata.get('title', 'N/A')}\n"
                    )
                    if 'file_path' in metadata:
                        source_info += f"檔案路徑: {metadata.get('file_path', 'N/A')}\n"
                    if 'file_type' in metadata:
                        source_info += f"檔案類型: {metadata.get('file_type', 'N/A')}\n"
                elif document_type == "paper":
                    # 論文格式：顯示論文資訊
                    authors = metadata.get('authors', [])
                    if isinstance(authors, str):
                        authors_str = authors
                    elif isinstance(authors, list):
                        authors_str = ', '.join(authors[:3])  # 最多顯示 3 個作者
                        if len(authors) > 3:
                            authors_str += f" 等 {len(authors)} 位作者"
                    else:
                        authors_str = 'N/A'
                    
                    source_info = (
                        f"[來源 {i}]\n"
                        f"論文標題: {metadata.get('title', 'N/A')}\n"
                        f"arXiv ID: {metadata.get('arxiv_id', 'N/A')}\n"
                        f"作者: {authors_str}\n"
                        f"發布日期: {metadata.get('published', 'N/A')}\n"
                    )
                else:
                    # 通用格式：顯示可用的資訊
                    source_info = f"[來源 {i}]\n"
                    if 'title' in metadata:
                        source_info += f"標題: {metadata.get('title', 'N/A')}\n"
                    if 'file_path' in metadata:
                        source_info += f"檔案: {metadata.get('file_path', 'N/A')}\n"
                    if 'arxiv_id' in metadata:
                        source_info += f"arXiv ID: {metadata.get('arxiv_id', 'N/A')}\n"
                
                # 添加相關性分數（如果有的話）
                rerank_score = result.get('rerank_score')
                hybrid_score = result.get('hybrid_score')
                if rerank_score is not None:
                    source_info += f"相關性分數: {rerank_score:.4f}\n"
                elif hybrid_score is not None:
                    source_info += f"相關性分數: {hybrid_score:.4f}\n"
                
                source_info += f"---\n{content}\n"
                formatted_parts.append(source_info)
                
            elif format_style == "simple":
                # 簡單格式：只包含關鍵資訊
                title = metadata.get('title', 'N/A')
                if document_type == "paper" and 'arxiv_id' in metadata:
                    arxiv_id = metadata.get('arxiv_id', 'N/A')
                    source_info = (
                        f"[來源 {i}: {title} "
                        f"(arXiv:{arxiv_id})]\n"
                        f"{content}\n"
                    )
                elif document_type == "cv" and 'file_path' in metadata:
                    file_path = metadata.get('file_path', 'N/A')
                    source_info = (
                        f"[來源 {i}: {title} "
                        f"({file_path})]\n"
                        f"{content}\n"
                    )
                else:
                    source_info = (
                        f"[來源 {i}: {title}]\n"
                        f"{content}\n"
                    )
                formatted_parts.append(source_info)
            else:  # minimal
                # 最小格式：只標註來源
                if document_type == "paper" and 'arxiv_id' in metadata:
                    arxiv_id = metadata.get('arxiv_id', 'N/A')
                    source_info = (
                        f"[arXiv:{arxiv_id}]\n"
                        f"{content}\n"
                    )
                elif 'title' in metadata:
                    title = metadata.get('title', 'N/A')
                    source_info = (
                        f"[{title}]\n"
                        f"{content}\n"
                    )
                else:
                    source_info = (
                        f"[來源 {i}]\n"
                        f"{content}\n"
                    )
                formatted_parts.append(source_info)
        
        formatted_text = "\n" + "="*60 + "\n".join(formatted_parts)
        
        # 如果設置了最大長度，進行截斷
        if self.max_context_length and len(formatted_text) > self.max_context_length:
            # 從後往前截斷，保留格式
            formatted_text = formatted_text[:self.max_context_length]
            # 確保最後一個來源資訊完整
            last_separator = formatted_text.rfind("="*60)
            if last_separator > 0:
                formatted_text = formatted_text[:last_separator] + "\n（內容已截斷...）"
        
        return formatted_text
    
    def create_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        document_type: str = "general"
    ) -> str:
        """
        創建完整的 LLM prompt
        
        Args:
            query: 用戶查詢
            context: 格式化後的上下文
            system_prompt: 可選的系統提示詞（如果為 None，會根據語言和文檔類型自動選擇）
            document_type: 文檔類型 ("paper", "cv", "general")
            
        Returns:
            完整的 prompt 字符串
        """
        # 自動檢測語言並選擇相應的系統提示詞
        if system_prompt is None and self.auto_detect_language:
            detected_language = self.detect_language(query)
            system_prompt = self.get_system_prompt(detected_language, document_type)
        elif system_prompt is None:
            # 如果禁用自動檢測，使用中文作為預設
            system_prompt = self.get_system_prompt("zh", document_type)
        
        # 根據檢測到的語言選擇提示詞格式
        detected_language = self.detect_language(query) if self.auto_detect_language else "zh"
        
        # 根據文檔類型選擇不同的提示詞結尾
        if document_type == "paper":
            if detected_language == "zh":
                ending = "## 請基於上述文獻片段回答問題，並在回答中引用具體的論文來源。"
            else:
                ending = "## Please answer the question based on the above document excerpts and cite specific paper sources in your answer."
        else:
            if detected_language == "zh":
                ending = "## 請基於上述文檔片段回答問題，並在回答中引用具體的文檔內容。"
            else:
                ending = "## Please answer the question based on the above document excerpts and cite specific document content in your answer."
        
        if detected_language == "zh":
            prompt = f"""{system_prompt}

## 相關文檔片段：

{context}

## 用戶問題：

{query}

{ending}"""
        else:  # English
            prompt = f"""{system_prompt}

## Relevant Document Excerpts:

{context}

## User Question:

{query}

{ending}"""
        
        return prompt
    
    def format_for_llm(
        self,
        query: str,
        results: List[Dict],
        system_prompt: Optional[str] = None,
        document_type: str = "general"
    ) -> str:
        """
        一站式方法：格式化檢索結果並創建完整的 prompt
        
        Args:
            query: 用戶查詢
            results: 檢索結果列表
            system_prompt: 可選的系統提示詞
            document_type: 文檔類型 ("paper", "cv", "general")
            
        Returns:
            完整的 prompt 字符串
        """
        context = self.format_context(results, document_type=document_type)
        return self.create_prompt(query, context, system_prompt, document_type)

