from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # 向量数据库
from PyPDF2 import PdfReader  # 读取pdf文件
from langchain.text_splitter import CharacterTextSplitter  # 文本分割器
from langchain_community.vectorstores import FAISS  # 向量库
from langchain_community.llms import QianfanLLMEndpoint  # 千帆大模型平台库
from langchain.llms import OpenAI  # openai模型库
import streamlit as st  # 搭建web界面
from langchain.chains import ConversationalRetrievalChain  # 对话检索链
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  #向量模型
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    doc_reader = PdfReader("./《城市轨道交通桥梁设计规范》（GBT51234-2017）.pdf")

    raw_text = ""
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", "。"]
    )

    texts = splitter.split_text(raw_text)

    # 定义向量模型路径
    EMBEDDING_MODEL = './m3e-base'

    # 第一步：加载文档：
    # loader = UnstructuredFileLoader('apiDocs.txt')
    # data = loader.load()
    # print(f'data-->{data}')
    # 第二步：切分文档：
    # text_split = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=4)
    # split_data = text_split.split_documents(data)
    # print(f'split_data-->{split_data}')

    # 第三步：初始化huggingface模型embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 第四步：将切分后的文档进行向量化，并且存储下来
    db = FAISS.from_texts(texts, embeddings)
    db.save_local('./faiss/standards')

    return texts


if __name__ == '__main__':
    texts = main()
    print(f'split_data-->{texts}')
