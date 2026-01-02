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
    
    # 適合 16GB MacBook Air 的模型推薦
    RECOMMENDED_MODELS = {
        "llama3.2:3b": {
            "name": "llama3.2:3b",
            "description": "Meta Llama 3.2 3B - 輕量級，適合 16GB 內存",
            "memory_required": "~4GB",
            "quality": "良好"
        },
        "llama3.2:1b": {
            "name": "llama3.2:1b",
            "description": "Meta Llama 3.2 1B - 極輕量級，快速響應",
            "memory_required": "~2GB",
            "quality": "基礎"
        },
        "phi3:mini": {
            "name": "phi3:mini",
            "description": "Microsoft Phi-3 Mini - 小模型，高質量",
            "memory_required": "~3GB",
            "quality": "良好"
        },
        "gemma:2b": {
            "name": "gemma:2b",
            "description": "Google Gemma 2B - 輕量級，開源",
            "memory_required": "~3GB",
            "quality": "良好"
        },
        "mistral:7b": {
            "name": "mistral:7b",
            "description": "Mistral 7B - 較大但質量高（如果內存足夠）",
            "memory_required": "~8GB",
            "quality": "優秀"
        }
    }
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        初始化 Ollama LLM
        
        Args:
            model_name: Ollama 模型名稱（預設: llama3.2:3b）
            base_url: Ollama API 基礎 URL
            timeout: 請求超時時間（秒）
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api"
        
        # 檢查模型是否在推薦列表中
        if model_name not in self.RECOMMENDED_MODELS:
            logger.warning(
                f"⚠️  模型 '{model_name}' 不在推薦列表中。"
                f"推薦的模型: {', '.join(self.RECOMMENDED_MODELS.keys())}"
            )
        
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
    
    @classmethod
    def print_recommended_models(cls):
        """打印推薦的模型列表"""
        print("\n" + "="*60)
        print("適合 16GB MacBook Air 的 Ollama 模型推薦")
        print("="*60)
        print()
        
        for model_key, info in cls.RECOMMENDED_MODELS.items():
            print(f"📦 {info['name']}")
            print(f"   描述: {info['description']}")
            print(f"   內存需求: {info['memory_required']}")
            print(f"   質量: {info['quality']}")
            print(f"   下載命令: ollama pull {info['name']}")
            print()

