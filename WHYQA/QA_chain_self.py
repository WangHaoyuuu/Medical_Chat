from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Chroma
import sys
from model_to_llm import init_llm
from init_db import init_db
import sys
import re
from langchain.embeddings.huggingface import HuggingFaceEmbeddings



class QA_chain_self():
    #基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    def __init__(self, 
                 model:str, 
                 temperature:float=0.36, 
                 appid:str=None,
                 api_key:str=None,
                 api_secret:str=None,
                 persist_path:str=None,
                 embedding = None,
                 template=default_template_rq ,
                 top_k:int=3
                 ):
        
        self.model = model
        self.temperature = temperature
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.embedding = embedding
        self.template = template
        self.top_k = top_k

        self.vectordb = init_db(persist_path=self.persist_path, 
                                embedding=self.embedding)
        
        print("vectordb init success!")

        self.llm = init_llm(self.model, 
                            self.temperature, 
                            self.appid, 
                            self.api_key, 
                            self.api_secret)
        
        print(f"{self.llm._llm_type} init success!")

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=self.template)
        

        self.retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': self.top_k})  #默认similarity，k=4
        print("retriever init success!")

        # 自定义 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        retriever=self.retriever,
                                        return_source_documents=False,
                                        chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})


    def answer(self, question:str=None, temperature = None, top_k = 4):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """
        if len(question) == 0:
            return ""
        if temperature == None:
            temperature = self.temperature
        result = self.qa_chain.invoke({"query": question, "temperature": temperature, "top_k": top_k})
        answer = result["result"]
        return answer   

# 测试
if __name__ == "__main__":
    embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    for model in ["Yi-34B-Chat", "Sparkv3", "glm-3-turbo"]:
        qa = QA_chain_self(model=model, embedding=embedding)
        result = qa.answer("什么是精神科疾病")
        print(result)
