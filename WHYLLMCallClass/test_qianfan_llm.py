# 向授权服务地址 https://aip.baidubce.com/oauth/2.0/token 发送请求（推荐使用POST）。
import requests
import json
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List, Optional, Any
import sys
import os                # 用于操作系统相关的操作，例如读取环境变量
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from test_MYLLM import CustomLLM
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForLLMRun
_ = load_dotenv(find_dotenv())


def get_access_token(API_KEY: str, SECRET_KEY: str):
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    headers = {"Content-Type": "application/json"}
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

class QianFanLLM(CustomLLM):
    model_name: str = None
    """
    model_name: 可选择
    - 
    """
    url: str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token={}"
    api_key: str = None
    appid: str = None
    api_secret: str = None
    access_token: str = None
    request_timeout: float = 50.0
    # model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    def init_access_token(self):
        # 当模型初始化时，获取access_token
        # 同时需要有api_key和api_secret
        if self.api_key is None or self.api_secret is None:
            raise ValueError("api_key and api_secret are required.")
        else:
            try:
                self.access_token = get_access_token(self.api_key, self.api_secret)
                # print("access_token: ", self.access_token)
            except Exception as e:
                print(e)
                raise  # 重新抛出异常
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 调用模型的逻辑
        if self.access_token is None:
            self.init_access_token()
        # 发送请求到api调用的url
        url = self.url.format(self.access_token)

        # 配置POST参数
        payload = json.dumps({
                "messages": [
                    {"role": "user", 
                    "content": prompt}
                    ],
                'temperature': self.temperature
        })


        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)
        # 返回结果
        # print(response)
        if response.status_code != 200:
            raise ValueError("Request failed with status code: {}".format(response.status_code))
        else:
            # 返回的是一个 Json 字符串
            text = json.loads(response.text)
            # print(js)
            return text["result"]
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **{"model": self.model_name},
            **super()._identifying_params,
        }
    
    @property
    def _llm_type(self) -> str:
        return "QianFanLLM"
            
if __name__ == "__main__":

    # 测试QianFanLLM
    model = QianFanLLM(model_name="Yi-34B-Chat", 
                       api_key="", 
                       api_secret="")
    result = model.invoke("你好")
    print(result)