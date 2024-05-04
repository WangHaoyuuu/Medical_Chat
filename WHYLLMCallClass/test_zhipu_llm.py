import requests
import json
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List, Optional, Any
from test_MYLLM import CustomLLM
import zhipuai
from langchain.pydantic_v1 import Field, root_validator
from langchain.utils import get_from_dict_or_env
from zhipuai import ZhipuAI
_ = load_dotenv(find_dotenv())

class ZhiPuLLM(CustomLLM):
    url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    model_name: str = None
    api_key: str = None
    appid: str = None
    api_secret: str = None
    access_token: str = None
    client: Any
    request_timeout: float = 50.0
    temperature: float = 0.36
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # 验证zhipu_api_key
    @root_validator
    def validate_api_key(cls, values: Dict) -> Dict:
        values["ZHIPUAI_API_KEY"] = get_from_dict_or_env(values,"zhipuai_api_key", "ZHIPUAI_API_KEY")

        try:
            zhipuai.api_key = values["ZHIPUAI_API_KEY"]
            values["client"] = ZhipuAI(api_key=zhipuai.api_key)
        except Exception as e:
            raise ValueError("api_key is required.")

        return values

    def _call(self, prompt: str, **kwargs: Any,) -> str:     
                # 配置POST参数
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                    {"role": "user", 
                    "content": prompt}
                    ],
                'temperature': self.temperature
        })
        # print(self.api_key)
        headers = {
            'Authorization': 'Bearer ' + self.api_key,
            "Content-Type": "application/json"
               }
        response = requests.request("POST", self.url, headers=headers, data=payload, timeout=self.request_timeout)
        # print(response)
        if response.status_code != 200:
            raise ValueError("Request failed with status code: {}".format(response.status_code))
        else:
            # 返回的是一个 Json 字符串
            text = json.loads(response.text)
            return text["choices"][0]["message"]["content"]
        

        # return response.choices[0].message.content
            
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **super()._identifying_params,
        }
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ZhiPuLLM"

if __name__ == "__main__":

    # # 测试ZhiPuLLM
    model = ZhiPuLLM(model_name = "glm-3-turbo",
                      api_key = "26128e545ddd8a44c6588c4d530a5fbe.hj6klSWNSFWnli9p",
                      )

    print(model.invoke("你好"))