"""
LLM 集成模組：使用 Ollama 進行本地 LLM 推理
"""
from typing import Optional, Dict, List
import logging
import requests
import json

logger = logging.getLogger(__name__)


class OllamaLLM:
    """使用 Ollama 進行本地 LLM 推理"""
    
    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        初始化 Ollama LLM
        
        Args:
            model_name: Ollama 模型名稱（預設: deepseek-r1:7b）
            base_url: Ollama API 基礎 URL
            timeout: 請求超時時間（秒）
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api"
        
        logger.info(f"✅ Ollama LLM 初始化完成 (模型: {model_name})")
    
    def _check_ollama_connection(self) -> bool:
        """
        檢查 Ollama 服務是否可用
        
        Returns:
            是否連接成功
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ 無法連接到 Ollama: {e}")
            logger.error(f"   請確保 Ollama 正在運行: ollama serve")
            return False
    
    def _check_model_available(self) -> bool:
        """
        檢查模型是否已下載
        
        Returns:
            模型是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                return any(self.model_name in name for name in model_names)
            return False
        except Exception as e:
            logger.error(f"❌ 檢查模型時出錯: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        生成回答
        
        Args:
            prompt: 輸入 prompt
            temperature: 溫度參數（0.0-1.0），控制隨機性
            max_tokens: 最大生成 token 數（None 表示使用模型預設）
            stream: 是否使用流式輸出
            
        Returns:
            生成的回答
        """
        # 檢查連接
        if not self._check_ollama_connection():
            raise ConnectionError(
                f"無法連接到 Ollama 服務 ({self.base_url})\n"
                f"請確保 Ollama 正在運行：\n"
                f"  1. 安裝 Ollama: https://ollama.ai\n"
                f"  2. 啟動服務: ollama serve\n"
                f"  3. 下載模型: ollama pull {self.model_name}"
            )
        
        # 檢查模型
        if not self._check_model_available():
            logger.warning(
                f"⚠️  模型 '{self.model_name}' 可能未下載。"
                f"請運行: ollama pull {self.model_name}"
            )
        
        # 準備請求參數
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            # 發送請求
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            
            if response.status_code != 200:
                error_msg = response.text
                raise RuntimeError(f"Ollama API 錯誤: {error_msg}")
            
            if stream:
                # 流式處理
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                chunk = data['response']
                                full_response += chunk
                                print(chunk, end='', flush=True)
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                print()  # 換行
                return full_response
            else:
                # 非流式處理
                data = response.json()
                return data.get('response', '')
                
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"請求超時（{self.timeout}秒）。"
                f"可以嘗試增加 timeout 或使用更小的模型。"
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"無法連接到 Ollama 服務。"
                f"請確保 Ollama 正在運行：ollama serve"
            )
        except Exception as e:
            logger.error(f"❌ 生成回答時出錯: {e}")
            raise
    
    def list_available_models(self) -> List[str]:
        """
        列出本地可用的模型
        
        Returns:
            可用模型名稱列表
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m.get('name', '') for m in models]
            return []
        except Exception as e:
            logger.error(f"❌ 獲取模型列表時出錯: {e}")
            return []
    

