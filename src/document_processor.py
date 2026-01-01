"""
文檔處理模組：載入 arXiv 論文並進行文字分割
"""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

