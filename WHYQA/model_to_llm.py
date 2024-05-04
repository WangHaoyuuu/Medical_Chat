import sys
import os                # 用于操作系统相关的操作，例如读取环境变量
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构造模块的绝对路径
module_dir0 = os.path.abspath(os.path.join(script_dir, "../"))
module_dir1 = os.path.abspath(os.path.join(script_dir, "../WHYLLMCallClass"))
# 添加模块的路径到 sys.path
sys.path.append(module_dir0)
sys.path.append(module_dir1)

from WHYLLMCallClass.test_qianfan_llm import QianFanLLM
from WHYLLMCallClass.test_call_llm import parse_api_key
from WHYLLMCallClass.test_spark_llm import Spark_LLM
from WHYLLMCallClass.test_zhipu_llm import ZhiPuLLM

def init_llm(model:str=None, 
             temperature:float=0.36,
             appid:str=None,
             api_key:str=None,
             api_secret:str=None):
    
    if model in ["Yi-34B-Chat"]:
        if api_key == None or api_secret == None:
            api_key, api_secret = parse_api_key("qianfan")
        llm = QianFanLLM(model_name = model, temperature=temperature, api_key=api_key, api_secret=api_secret)

    elif model in ["Sparkv3"]:
        if appid == None or api_key == None or api_secret == None:
            appid, api_key, api_secret = parse_api_key("spark")
        llm = Spark_LLM(appid=appid, api_key=api_key, api_secret=api_secret)

    elif model in ["glm-3-turbo"]:
        if api_key == None:
            api_key = parse_api_key("zhipu")
        llm = ZhiPuLLM(model_name = model, api_key=api_key)

    else:
        return ValueError(f"model{model} not support!!!")
    return llm


def get_completion(model:str=None,
                   input:str=None):
    
    llm = init_llm(model=model)
    return llm.invoke(input)


if __name__ == "__main__":
    # model = init_llm(model="Yi-34B-Chat")
    # result = model.invoke(input="你好")
    # print(result)

    # model = init_llm(model = "glm-3-turbo")
    # result = model.invoke("你好")
    # print(result)

    # model = init_llm(model = "Sparkv3")
    # result = model.invoke("你好，你是谁")
    # print(result)
    model = "Sparkv3"
    result = get_completion(model=model, input="你好, 你是谁？")
    print(result)