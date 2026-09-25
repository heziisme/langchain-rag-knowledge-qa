"""
RAG本地知识库问答系统【修复版：修正距离判断逻辑】
Chroma similarity_search_with_score 返回为欧氏距离：分数越小越相似
功能：读取docs目录下md/txt文档 → 文本分割 → Chroma向量入库 → 用户提问召回片段 → LLM生成回答
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# ===================== 配置项 =====================
DOC_FOLDER = "./docs"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
RETRIEVE_TOP_K = 5
# Chroma欧氏距离，根据你实际输出分数，大于该值视为不相关，直接丢弃
SIMILARITY_THRESHOLD = 1.3

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 构建prompt模板
RAG_PROMPT_TEMPLATE = """
你是知识库问答助手，请严格根据下面【参考上下文】回答用户问题。
如果上下文没有相关信息，直接回答：「知识库中没有找到相关信息」，禁止编造内容。

【参考上下文】
{context}

【用户问题】
{question}

回答：
"""
prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def load_all_docs(folder_path: str):
    """读取文件夹下全部md、txt文档"""
    docs_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".md", ".txt")):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            doc = loader.load()
            docs_list.extend(doc)
    print(f"✅成功读取文档数量：{len(docs_list)}")
    return docs_list

def build_vector_store():
    """加载文档，切分文本，构建Chroma向量库"""
    raw_docs = load_all_docs(DOC_FOLDER)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"✅文档切分后片段总数：{len(split_docs)}")

    vector_store = Chroma.from_documents(
        documents=split_docs,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("✅向量数据库构建完成，已保存到chroma_db文件夹")
    return vector_store

def get_vector_store():
    """获取向量库，没有则重新构建"""
    if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR))>0:
        print("✅加载已存在的向量数据库")
        vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR)
    else:
        print("⚠️未检测到向量库，开始构建向量库……")
        vector_store = build_vector_store()
    return vector_store

def rag_answer(question: str):
    vector_store = get_vector_store()
    search_docs_with_score = vector_store.similarity_search_with_score(question, k=RETRIEVE_TOP_K)

    print("\n----------【检索到的知识库片段(欧氏距离，越小越相似)】----------")
    valid_docs = []
    for idx, (doc, score) in enumerate(search_docs_with_score):
        print(f"片段{idx+1} 距离={score:.3f}：{doc.page_content[:150]}……\n")
        # ==========修复：距离小于阈值才保留，距离越大越不相关==========
        if score < SIMILARITY_THRESHOLD:
            valid_docs.append(doc)

    if len(valid_docs) == 0:
        context_text = ""
        print("⚠️经过阈值过滤，没有保留有效知识库片段")
    else:
        context_text = "\n".join([d.page_content for d in valid_docs])

    llm = ChatDeepSeek(model="deepseek-chat", api_key=api_key)
    chain = prompt | llm
    resp = chain.invoke({"context": context_text, "question": question})
    return resp.content

if __name__ == "__main__":
    print("===== 本地RAG知识库问答系统 =====")
    print("提示：文档放在 ./docs 目录，输入 q 退出程序")
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() == "q":
            print("程序退出")
            break
        ans = rag_answer(user_input)
        print(f"\n🤖RAG回答：{ans}")
