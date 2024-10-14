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

# from project2.API_Config import *
os.environ["QIANFAN_AK"] = "KmkF2XaSExnW2DrDjrO1M4q0"
os.environ["QIANFAN_SK"] = "VhZXP8AI8a98BuWnqrvTwuS2J648jAtc"

# 使用国外OPENAI的模型，需要导入API-KEY(需要科学上网)
# os.environ[ "OPENAI_API_KEY" ] = OPENAI_AK
# 使用国内百度千帆平台的模型，需要导入API-KEY和SERECT-KEY
# 设置web页面：比如标题、描述功能
st.title("《城市规范查询》")
st.write("请进行查询")

# embeddings模型
embeddings = HuggingFaceEmbeddings(model_name=r"C:\codes\demonstration\01-code\RAG\m3e-base")
# 创建文档搜索
# docsearch = FAISS.from_embeddings()
db = FAISS.load_local(r'C:\codes\demonstration\01-code\RAG\faiss\standards'
                      , embeddings, allow_dangerous_deserialization=True)

# 创建对话链
qa = ConversationalRetrievalChain.from_llm(
    llm=QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-7B', temperature=1, top_p=0.5),
    # llm=OpenAI(model='gpt-3.5-turbo'),
    retriever=db.as_retriever(), chain_type='refine',
    return_source_documents=True, )
# 初始化聊天记录列表
chat_history = []
# 获取用户的查询
query = st.text_input("请给出你的问题")
#添加一个生成按钮
generate_button = st.button("生成答案")
if generate_button and query:
    with st.spinner("答案生成中..."):
        # 将问题以及历史对话记录传入对话链获得模型输出结果
        result = qa({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        source_documents = result['source_documents']
        #将答案和source_documents合并为单个响应（输出）
        response = {"answer": answer, "source_documents": source_documents}
        st.write("response:", response)
