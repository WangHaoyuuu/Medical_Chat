import errno
import os
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

DEFAULT_DB_PATH = "/home/why/CODES/Medical_Chat/WHYmedicalbooks"
DEFAULT_PERSIST_PATH = "/home/why/CODES/Medical_Chat/WHYembedding"

def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('.pdf'):
                file_list.append(os.path.join(filepath, filename))
    return file_list

# Define the pattern to remove newline characters not surrounded by Chinese characters
# pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)

def clean_pdf_content(page_content):
    # Remove all newline characters
    content = page_content.replace('\n', '')
    # Remove all periods, including sequences of periods
    content = re.sub(r'\.+', '', content)
    # content = re.sub(r'\.+', '', content)
    return content

class PageContent:
    def __init__(self, content, metadata={}):
        self.page_content = content
        self.metadata = metadata

def create_db(files):
    if files is None:
        return "Can't load empty file list."
    if not isinstance(files, list):
        files = [files]

    for file_path in files:
        print(f"Processing file: {file_path}")
        loader = PyMuPDFLoader(file_path)
        pdf_pages = loader.load()

        cleaned_pages = [clean_pdf_content(page.page_content) for page in pdf_pages]

        print(f"Cleaned pages: {cleaned_pages[:5]}")

        # return


        # Wrap the cleaned page content in the expected object format, with default empty dictionary for metadata
        page_objects = [PageContent(content, {}) for content in cleaned_pages]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        split_docs = []
        for page_object in page_objects:
            # Assuming split_documents now receives the correct input format
            split_docs.extend(text_splitter.split_documents([page_object]))
        # Get embeddings
        embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

        # Create a unique persistence path for each file's parent directory
        parent_dir_name = Path(file_path).parent.stem
        persist_dir = os.path.join(DEFAULT_PERSIST_PATH, parent_dir_name, "vector_db")
        
        print(f"Persistence path: {persist_dir}")
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)
            
        # Create a vector database
        vectordb = Chroma.from_documents(
            
            documents=split_docs[:],  # Assuming you want to process the first 10 chunks
            embedding=embedding,
            persist_directory=persist_dir
        )
        print(f"Vector database created: {vectordb}")
        vectordb.persist()

def load_knowledge_db(path, embeddings):
    """ 

    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数 据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。  

    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb



if __name__ == "__main__":
    pdf_files = get_files(DEFAULT_DB_PATH)
    print(f"Found {len(pdf_files)} PDF files.")
    create_db(files=pdf_files)