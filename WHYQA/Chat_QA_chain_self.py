from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import sys
from model_to_llm import init_llm
from init_db import init_db
import sys
import re
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain

class Chat_QA_chain_self:
    # default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    # 案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    # {context}
    # 问题: {question}
    # 有用的回答:"""

    def __init__(self, 
                 model:str, 
                 temperature:float=0.36, 
                 appid:str=None,
                 api_key:str=None,
                 api_secret:str=None,
                 persist_path:str=None,
                 embedding = None,
                 top_k:int=3,
                 chat_history:list=[],
                 ):
        self.model = model
        self.temperature = temperature
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.embedding = embedding
        self.chat_history = chat_history
        self.top_k = top_k

        self.vectordb = init_db(persist_path=self.persist_path,
                                embedding=self.embedding)
        print("vectordb init success!")

        self.retriever = self.vectordb.as_retriever(search_type="similarity",
                                                    search_kwargs={'k': self.top_k})
        
        print("retriever init success!")

        self.llm = init_llm(self.model,
                            self.temperature,
                            self.appid,
                            self.api_key,
                            self.api_secret)
        print(f"{self.llm._llm_type} init success!")

        self.chat_qa_chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever = self.retriever,
        )

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()
    
    def change_history_length(self,history_len:int=5):
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

    def answer(self, question:str=None, temperature = None, top_k = 4):
        if temperature is None:
            temperature = self.temperature
        if top_k is None:
            top_k = self.top_k
        
        result = self.chat_qa_chain.invoke({"question": question, "chat_history": self.chat_history})
        # print(result)
        answer = result['answer']
        answer = re.sub(r"\n", "", answer)
        answer = re.sub(r"\n\n", "", answer)
        self.chat_history.append((question,answer))
        # print(self.chat_history)
        return answer
                                
# 测试
if __name__ == "__main__":
    embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    for model in ["Sparkv3"]:
        qa = Chat_QA_chain_self(model=model, embedding=embedding)
        result = qa.answer("什么是精神科疾病")
        print(result)
        result = qa.answer("精神科疾病有哪些")
        print(result)