from typing import Any, Dict, Mapping
from langchain_core.language_models.llms import LLM
from typing import Dict, Any
from pydantic import Field
class CustomLLM(LLM):
    """自定义语言模型类，继承自LLM基类。

    属性:
        model_name: 模型名称。
        url: 模型平台的URL，需要先去某个url获得access_token,然后再去另一个url获得模型的结果。（可选）
        api_key: API密钥。（可选）
        appid: 应用ID。（可选）
        api_secret: API密码。（可选）
        access_token: 模型平台的访问令牌。（可选）
        timeout: 模型平台的超时时间。
        temperature: 控制生成文本的随机性的参数，值越大，输出的随机性越大。
        model_kwargs: 模型的其他可选参数。
    """

    url: str = None

    model_name: str = None

    api_key: str = None

    appid: str = None

    api_secret: str = None

    access_token: str = None

    client: Any

    request_timeout: float = 50.0

    temperature: float = 0.36

    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # 定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}