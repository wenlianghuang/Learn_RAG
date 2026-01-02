"""
文檔處理模組：載入 arXiv 論文並進行文字分割
支援本地文件：PDF, DOCX, TXT
"""
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
import arxiv


class DocumentProcessor:
    """處理 arXiv 論文文檔，進行分割和準備"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文檔處理器
        
        Args:
            chunk_size: 每個 chunk 的大小（字符數）
            chunk_overlap: chunk 之間的重疊大小
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
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
            full_text = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
            
            # 分割文字
            chunks = self.text_splitter.split_text(full_text)
            
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
        
        # 分割文字
        chunks = self.text_splitter.split_text(file_doc["content"])
        
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

