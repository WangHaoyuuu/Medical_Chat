import sys
import os                # 用于操作系统相关的操作，例如读取环境变量
import sys

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构造模块的绝对路径
module_dir0 = os.path.abspath(os.path.join(script_dir, "../"))
module_dir1 = os.path.abspath(os.path.join(script_dir, "../WHYLLMCallClass"))
# 添加模块的路径到 sys.path
sys.path.append(module_dir0)
sys.path.append(module_dir1)



# print(sys.path)

from WHYLLMCallClass import test_qianfan_llm
llm = test_qianfan_llm.QianFanLLM(model_name="Yi-34B-Chat", 
                       api_key="", 
                       api_secret="")

result = llm.invoke("你好")
print(result)