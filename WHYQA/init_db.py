import sys

from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def init_db(persist_path = None, 
            embedding = None):
    vectordb = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding
    )
    # vectordb.persist() 
    vectordb.persist()
    return vectordb

# 测试
if __name__ == "__main__":
    persist_path = '/workspaces/Medical_Chat/WHYembedding/精神科/vector_db'
    embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

    vector_db = init_db(persist_path=persist_path, embedding=embedding)
    print(vector_db._collection.count())
    
    question = "什么是抑郁症"
    docs = vector_db.similarity_search(question,k=3)
    print(f"检索到的内容数：{len(docs)}")