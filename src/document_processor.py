"""
文檔處理模組：載入 arXiv 論文並進行文字分割
支援本地文件：PDF, DOCX, TXT
支援兩種分塊策略：
1. 字符分塊（預設）：基於固定字符數的分塊，速度快
2. 語義分塊（可選）：基於語義相似度的分塊，能保持語義完整性
"""
from typing import List, Dict, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
import arxiv

# 嘗試導入語義分塊器（需要 langchain-experimental）
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False


class DocumentProcessor:
    """
    處理 arXiv 論文文檔，進行分割和準備
    
    支援兩種分塊模式：
    - 字符分塊（預設）：快速、穩定，適合大多數場景
    - 語義分塊（可選）：更智能，能保持語義完整性，但需要額外依賴和計算時間
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        embeddings: Optional[Any] = None,  # 可選：用於語義分塊的 embedding 模型
        use_semantic_chunking: bool = False,  # 是否使用語義分塊
        breakpoint_threshold_amount: float = 1.5,  # 語義分塊敏感度（標準差倍數）
        min_chunk_size: int = 100  # 語義分塊的最小 chunk 大小（字符數）
    ):
        """
        初始化文檔處理器
        
        Args:
            chunk_size: 每個 chunk 的大小（字符數），僅用於字符分塊模式
            chunk_overlap: chunk 之間的重疊大小（字符數），僅用於字符分塊模式
            embeddings: 用於計算語義距離的 embedding 模型物件（可選）
                       當 use_semantic_chunking=True 時必須提供
            use_semantic_chunking: 是否使用語義分塊
                                  True: 使用語義分塊（需要提供 embeddings）
                                  False: 使用字符分塊（預設）
            breakpoint_threshold_amount: 語義分塊的敏感度參數
                                        數值越大，分塊越少（chunks 越大）
                                        數值越小，分塊越多（chunks 越小）
                                        建議範圍：1.0 - 2.0，預設 1.5
            min_chunk_size: 語義分塊的最小 chunk 大小（字符數）
                           小於此大小的 chunks 會被合併到相鄰的 chunks
                           預設 100 字符
        """
        self.embeddings = embeddings
        self.use_semantic_chunking = use_semantic_chunking
        self.min_chunk_size = min_chunk_size
        
        # 如果要求使用語義分塊
        if use_semantic_chunking:
            # 檢查是否安裝了必要的依賴
            if not SEMANTIC_CHUNKER_AVAILABLE:
                raise ImportError(
                    "使用語義分塊需要安裝 langchain-experimental 套件。\n"
                    "請執行: pip install langchain-experimental\n"
                    "或使用字符分塊模式（use_semantic_chunking=False）"
                )
            
            # 檢查是否提供了 embeddings
            if embeddings is None:
                raise ValueError(
                    "使用語義分塊時必須提供 embeddings 參數。\n"
                    "範例：\n"
                    "  from langchain_community.embeddings import HuggingFaceEmbeddings\n"
                    "  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')\n"
                    "  processor = DocumentProcessor(embeddings=embeddings, use_semantic_chunking=True)"
                )
            
            # 初始化語義分塊器
            # 使用「標準差」策略：當相鄰句子之間的語義距離超過平均距離的標準差倍數時，進行切分
            self.text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="standard_deviation",
                breakpoint_threshold_amount=breakpoint_threshold_amount
            )
            print(f"✓ 使用語義分塊模式（敏感度: {breakpoint_threshold_amount}，最小 chunk 大小: {min_chunk_size} 字符）")
        else:
            # 使用傳統的字符分塊（預設模式）
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            print(f"✓ 使用字符分塊模式（大小: {chunk_size} 字符，重疊: {chunk_overlap} 字符）")
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        後處理 chunks：過濾和合併太小的 chunks
        
        語義分塊可能會產生一些非常小的 chunks（例如只有幾個單詞），
        這些小 chunks 可能不包含足夠的上下文信息。此方法會：
        1. 將小於 min_chunk_size 的 chunks 合併到相鄰的 chunks
        2. 確保最終的 chunks 都有足夠的大小
        
        Args:
            chunks: 原始 chunks 列表（從分塊器產生的）
            
        Returns:
            處理後的 chunks 列表（過濾和合併後的）
        """
        # 如果使用字符分塊，不需要後處理（因為已經有固定大小）
        if not self.use_semantic_chunking:
            return chunks
        
        # 如果沒有 chunks，直接返回
        if not chunks:
            return chunks
        
        processed = []
        current_small_chunk = ""  # 累積的小 chunk
        
        for chunk in chunks:
            chunk_stripped = chunk.strip()
            chunk_length = len(chunk_stripped)
            
            # 如果當前 chunk 太小，嘗試與下一個合併
            if chunk_length < self.min_chunk_size:
                # 累積到臨時變數中
                if current_small_chunk:
                    current_small_chunk += "\n\n" + chunk
                else:
                    current_small_chunk = chunk
            else:
                # 當前 chunk 足夠大
                # 如果有累積的小 chunk，先處理它
                if current_small_chunk:
                    current_small_chunk_stripped = current_small_chunk.strip()
                    if len(current_small_chunk_stripped) >= self.min_chunk_size:
                        # 累積後足夠大，作為獨立 chunk
                        processed.append(current_small_chunk)
                    else:
                        # 累積後還是太小，合併到上一個 chunk（如果存在）
                        if processed:
                            processed[-1] += "\n\n" + current_small_chunk
                        else:
                            # 如果沒有上一個 chunk，還是要保留
                            processed.append(current_small_chunk)
                    current_small_chunk = ""
                
                # 添加當前足夠大的 chunk
                processed.append(chunk)
        
        # 處理最後的累積小 chunk
        if current_small_chunk:
            current_small_chunk_stripped = current_small_chunk.strip()
            if len(current_small_chunk_stripped) >= self.min_chunk_size:
                # 足夠大，作為獨立 chunk
                processed.append(current_small_chunk)
            elif processed:
                # 太小，合併到最後一個 chunk
                processed[-1] += "\n\n" + current_small_chunk
            else:
                # 如果沒有其他 chunks，還是要保留
                processed.append(current_small_chunk)
        
        return processed
    
    def fetch_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        從 arXiv 獲取論文
        
        Args:
            query: 搜尋查詢（例如 "cat:cs.AI"）
            max_results: 最大結果數量
            
        Returns:
            論文列表，每個論文包含標題、摘要等資訊
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for paper in search.results():
            papers.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": str(paper.published),
                "arxiv_id": paper.entry_id.split('/')[-1],
                "arxiv_url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "categories": paper.categories,
            })
        
        return papers
    
    def process_documents(self, papers: List[Dict]) -> List[Dict]:
        """
        處理論文，將每篇論文分割成 chunks
        
        Args:
            papers: 論文列表
            
        Returns:
            處理後的文檔 chunks，每個 chunk 包含內容和元數據
        """
        documents = []
        
        for paper in papers:
            # 組合論文的完整文字（標題 + 摘要）
            # 保留換行符號 \n\n 作為語義斷點的結構參考
            full_text = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
            
            # 分割文字（根據選擇的模式：字符分塊或語義分塊）
            chunks = self.text_splitter.split_text(full_text)
            
            # 後處理：過濾和合併太小的 chunks（僅語義分塊模式）
            chunks = self._post_process_chunks(chunks)
            
            # 為每個 chunk 創建文檔物件
            for i, chunk in enumerate(chunks):
                doc = {
                    "content": chunk,
                    "metadata": {
                        "title": paper['title'],
                        "arxiv_id": paper['arxiv_id'],
                        "arxiv_url": paper['arxiv_url'],
                        "pdf_url": paper['pdf_url'],
                        "authors": paper['authors'],
                        "published": paper['published'],
                        "categories": paper['categories'],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunking_method": "semantic" if self.use_semantic_chunking else "character"
                    }
                }
                documents.append(doc)
        
        return documents
    
    def get_texts_and_metadatas(self, documents: List[Dict]):
        """
        從文檔列表中提取文字和元數據
        
        Args:
            documents: 文檔列表
            
        Returns:
            (texts, metadatas) 元組
        """
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        return texts, metadatas
    
    def load_from_file(self, file_path: str) -> Dict:
        """
        從本地文件載入文檔（支援 PDF, DOCX, TXT 等）
        
        Args:
            file_path: 文件路徑
            
        Returns:
            文檔字典，包含內容和元數據
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = file_path.suffix.lower()
        file_name = file_path.stem
        file_size = os.path.getsize(file_path)
        
        # 根據文件類型選擇不同的加載器
        if file_ext == '.pdf':
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                # 合併所有頁面
                full_text = "\n\n".join([page.page_content for page in pages])
            except ImportError:
                raise ImportError(
                    "需要安裝 pypdf 來處理 PDF 文件: pip install pypdf"
                )
        
        elif file_ext in ['.docx', '.doc']:
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(str(file_path))
                pages = loader.load()
                full_text = "\n\n".join([page.page_content for page in pages])
            except ImportError:
                raise ImportError(
                    "需要安裝 docx2txt 來處理 DOCX 文件: pip install docx2txt"
                )
        
        elif file_ext == '.txt':
            # 嘗試不同的編碼
            encodings = ['utf-8', 'gbk', 'big5', 'latin-1']
            full_text = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        full_text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if full_text is None:
                raise ValueError(f"無法讀取文件，嘗試的編碼都不適用: {encodings}")
        
        else:
            raise ValueError(
                f"不支援的文件類型: {file_ext}\n"
                f"支援的格式: .pdf, .docx, .doc, .txt"
            )
        
        if not full_text or len(full_text.strip()) == 0:
            raise ValueError(f"文件為空或無法提取文字: {file_path}")
        
        return {
            "title": file_name,
            "content": full_text,
            "file_path": str(file_path),
            "file_type": file_ext,
            "file_size": file_size,
        }
    
    def process_file(self, file_path: str) -> List[Dict]:
        """
        處理單個文件，分割成 chunks
        
        Args:
            file_path: 文件路徑
            
        Returns:
            處理後的文檔 chunks 列表
        """
        # 載入文件
        file_doc = self.load_from_file(file_path)
        
        # 分割文字（根據選擇的模式：字符分塊或語義分塊）
        chunks = self.text_splitter.split_text(file_doc["content"])
        
        # 後處理：過濾和合併太小的 chunks（僅語義分塊模式）
        chunks = self._post_process_chunks(chunks)
        
        if not chunks:
            raise ValueError(f"文件分割後沒有內容: {file_path}")
        
        # 創建文檔 chunks
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "content": chunk,
                "metadata": {
                    "title": file_doc["title"],
                    "file_path": file_doc["file_path"],
                    "file_type": file_doc["file_type"],
                    "file_size": file_doc["file_size"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunking_method": "semantic" if self.use_semantic_chunking else "character"
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_files(self, file_paths: List[str]) -> List[Dict]:
        """
        處理多個文件
        
        Args:
            file_paths: 文件路徑列表
            
        Returns:
            所有文件的文檔 chunks 列表
        """
        all_documents = []
        for file_path in file_paths:
            try:
                print(f"處理文件: {file_path}")
                documents = self.process_file(file_path)
                all_documents.extend(documents)
                print(f"  ✓ 創建了 {len(documents)} 個 chunks")
            except Exception as e:
                print(f"  ✗ 處理文件失敗: {file_path}")
                print(f"    錯誤: {e}")
                continue
        
        return all_documents

