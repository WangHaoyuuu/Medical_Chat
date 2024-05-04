import openai
import json
import requests
import _thread as thread
import base64
import datetime
from dotenv import load_dotenv, find_dotenv
import hashlib
import hmac
import os
import queue
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import zhipuai
from langchain.utils import get_from_dict_or_env
import websocket  # 使用websocket_client
import sys
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构造模块的绝对路径
module_dir0 = os.path.abspath(os.path.join(script_dir, "../"))
module_dir1 = os.path.abspath(os.path.join(script_dir, "../WHYQA"))
# 添加模块的路径到 sys.path
sys.path.append(module_dir0)
sys.path.append(module_dir1)

def parse_api_key(model, env_file : dict = None):
    """
    通过 model 和 env_file 的来解析平台参数
    """   
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "zhipu":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")

    elif model == "spark":
        return env_file["SPARK_APPID"], env_file["SPARK_API_KEY"], env_file["SPARK_API_SECRET"]

    elif model == "qianfan":
        return env_file["QIANFAN_AK"], env_file["QIANFAN_SK"]

    else:
        raise ValueError(f"model{model} not support!!!")

if __name__ == "__main__":
    print(sys.path)


