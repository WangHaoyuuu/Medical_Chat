import sys 
import sys 
import os
import os               
import re
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构造模块的相对路径
module_dirs = ["../", "../WHYLLMCallClass", "../WHYQA"]
# 添加模块的路径到 sys.path
sys.path.extend([os.path.abspath(os.path.join(script_dir, module_dir)) for module_dir in module_dirs])

#print(sys.path)

from WHYQA.model_to_llm import get_completion
from WHYQA.QA_chain_self import QA_chain_self
from WHYQA.Chat_QA_chain_self import Chat_QA_chain_self
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# LLM模型选择
LLM_MODEL_DICT = {
    "QIANFAN":["Yi-34B-Chat"],
    "SPARK":["Sparkv3"],
    "ZHIPU":["glm-3-turbo",]
}

EMBEDDING_MODEL_LIST = ['m3e']
INIT_EMBEDDING_MODEL = 'm3e'
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_MODEL = LLM_MODEL_LIST[0]
# print(INIT_MODEL)
# print(LLM_MODEL_LIST)
# 定义所有可选的向量数据库路径

embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

class qa_chain():
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self,
                                model:str, 
                                embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"),
                                temperature:float=0.36, 
                                question:str=None,
                                top_k:int=3,
                                chat_history:list=[],
                                department:str=None,
):
        """
        调用带历史记录的问答链进行回答
        """
        persist_path = db_paths[department]
        # persist_path = selected_db_key
        if question == None or len(question) == 0:
            # 测试
            print("请输入问题")
            return "请输入问题", chat_history
        try:
            key = (model, id(embedding))
            # 测试

            print(key)

            print(chat_history)
            # Unsupported chat history format: <class 'list'>. Full chat history:
            if chat_history == None or len(chat_history) == 0:
                chat_history = []
                print("chat_history is None")
            else:
                # chat_history示例
                # 将chat_history中的内容转换为元组而不是
                chat_history = [(item[0], item[1]) for item in chat_history]
            
                print(chat_history)
            if key not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[key] = Chat_QA_chain_self(model=model, 
                                                                   temperature=temperature,
                                                                   top_k=top_k, 
                                                                   chat_history=chat_history, 
                                                                   persist_path=persist_path, 
                                                                   embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"))
            # 打印persist_path
            print(persist_path)
            chain = self.chat_qa_chain_self[key]

            print(type(embedding))  # 应该输出 <class 'langchain.embeddings.huggingface.HuggingFaceEmbeddings'
            
            print(chain)
            # 假设 chain.answer 返回的是一个字符串
            
            # print(answer)
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            print(e)
            return [("Error", str(e))], chat_history


    
    def qa_chain_self_answer(self,
                             model:str, 
                             embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"),
                             temperature:float=0.36,                              
                             question:str=None,
                             top_k:int=3,
                             chat_history:list=[],
                             department:str=None,
                                ):
                            
                             
        """
        调用不带历史记录的问答链进行回答
        """ 
        persist_path = db_paths[department]
        if question == None or len(question) < 1:
            return "", chat_history  
        try:
            key = (model, id(embedding))
            print(key)
            if key not in self.qa_chain_self:
                print("初始化问答链")
                self.qa_chain_self[key] = QA_chain_self(model=model, 
                                                        temperature=temperature,
                                                        top_k=top_k, 
                                                        persist_path=persist_path, 
                                                        embedding=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"))
                print("问答链初始化成功") 
            # 打印persist_path
            print(persist_path)
            chain = self.qa_chain_self[key]
            answer = chain.answer(question=question, temperature=temperature, top_k=top_k)
            print(answer)
            chat_history.append(("User: " + question, "Assistant: " + answer))
            return "", chat_history
        except Exception as e:
            return [("Error", str(e))], chat_history

    # clear_history 还未测试
    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            print("清空历史记录")
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()



def format_chat_prompt(message:str, chat_history:list=[]):
    """
    格式化聊天prompt
    """
    prompt = ""
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    print(prompt)
    # 返回格式化后的 prompt。
    return prompt

def respond(message,
            chat_history,
            model,
            history_len=3,
            temperature=0.36,
            max_tokens=2048
            ):
    """ 
    生成普通的llm的回复
    """
    if message == None or len(message) < 1:
        return "请输入问题", chat_history
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(model, formatted_prompt)
        # 将bot_message中\n换为<br/>
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        bot_message = re.sub(r"\n", '<br/>', bot_message)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        return "", chat_history
    except Exception as e:
        return e, chat_history


example_prompts = [
    "请问什么是精神病？",
    "介绍一下深度学习的基本原理。",
    "如何使用Python进行数据分析？"
]

db_paths = {
    "儿科": '/home/why/CODES/Medical_Chat/WHYembedding/儿科/vector_db',
    "耳鼻喉科": '/home/why/CODES/Medical_Chat/WHYembedding/耳鼻喉科/vector_db',
    "妇产科": '/home/why/CODES/Medical_Chat/WHYembedding/妇产科/vector_db',
    "感染科": '/home/why/CODES/Medical_Chat/WHYembedding/感染科/vector_db',
    "内科": '/home/why/CODES/Medical_Chat/WHYembedding/内科/vector_db',
    "神经科": '/home/why/CODES/Medical_Chat/WHYembedding/神经科/vector_db',
    "外科": '/home/why/CODES/Medical_Chat/WHYembedding/外科/vector_db',
}

model_center = qa_chain()
import gradio as gr

block = gr.Blocks()
def update_msg(prompt):
    """Function to update the message box with the selected prompt."""
    return prompt

with block as demo:
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True)
            # msg的默认值是example_prompts[0]，即第一个例子。
            msg = gr.Textbox(label="Prompt")
            # msg = str(msg)
            db_select = gr.Dropdown(list(db_paths.keys()), label="选择数据库")
    
            example_prompt_table = gr.Radio(
                example_prompts, 
                label="Example Prompts",
                value=None, 
                interactive=True
            )
            example_prompt_table.change(update_msg, inputs=[example_prompt_table], outputs=[msg])

            with gr.Row():
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="Clear console")
        with gr.Column(scale=1):
            model_argument = gr.Accordion("参数配置", open=True)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)
                top_k = gr.Slider(1,
                                  5,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)
                
                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)
                
            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_MODEL,
                    interactive=True)
                embeddings = gr.Dropdown(
                    EMBEDDING_MODEL_LIST,
                    label="embeddings",
                    value=INIT_EMBEDDING_MODEL,
                    interactive=True)
                
        db_with_his_btn.click(
            model_center.chat_qa_chain_self_answer, 
            inputs=[
                llm, 
                embeddings, 
                temperature, 
                msg, 
                top_k, 
                chatbot, 
                db_select
            ],
            outputs=[msg, chatbot]
        )
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, 
                                inputs=[
                                    llm,         # model
                                    embeddings,  # embedding
                                    temperature, # temperature
                                    msg,         # question
                                    top_k,       # top_k
                                    chatbot,     # chat_history (这个参数应该是由 Chatbot 组件提供的历史记录)
                                    db_select
                                ],
                              outputs=[msg, chatbot])
        # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        llm_btn.click(respond, 
                            inputs=[
                            msg, 
                            chatbot,  
                            llm, 
                            history_len, 
                            temperature
                            ], 
                            outputs=[msg, chatbot],
                            show_progress="minimal")
        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(respond, inputs=[
                                msg, 
                                chatbot, 
                                llm, 
                                history_len, 
                                temperature], 
                                outputs=[msg, chatbot], 
                                show_progress="hidden")
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(share=True)