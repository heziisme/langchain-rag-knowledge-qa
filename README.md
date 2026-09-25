# 本地文档RAG知识库问答系统
项目简介：实现基于LangChain+Chroma向量数据库的RAG检索增强生成应用。
加载本地Markdown/TXT私有文档，做文本切分、向量化存储；用户提问时语义检索相关文档片段，交给大模型生成回答，限制模型不能编造知识库以外的内容。

## 技术栈
Python
LangChain（文档加载、文本分割、prompt编排）
Chroma 本地向量数据库
DeepSeek‑Chat LLM
python‑dotenv

## 项目目录结构
agent_demo_03
├─ .env                    # API 密钥配置
├─ .gitignore              # git 忽略配置
├─ rag_main.py             # RAG 主程序
├─ README.md
└─ docs/                   # 知识库文档目录，放置 md/txt
└─ sample_game_info.md # 示例文档

## 快速启动
1. 安装依赖
pip install langchain langchain-deepseek langchain-chroma langchain-community python-dotenv
2. 在.env填入DeepSeek API密钥
DEEPSEEK_API_KEY = 你的密钥
3. 将私有知识库md/txt文档放入`docs`文件夹
4. 运行主程序
python rag_main.py
> 第一次运行自动构建向量数据库，生成chroma_db文件夹。
> 如果新增/修改知识库文档，请删除`chroma_db`文件夹，重新运行程序重新向量化。

## 程序交互
- 在控制台输入问题进行知识库问答
- 输入 `q` 退出程序
- 检索会打印召回的知识库片段，方便调试RAG检索效果
- 知识库无相关内容时，模型会如实告知，禁止幻觉编造

## 核心功能点
1. 本地文档加载，支持md/txt格式
2. 递归文本分割，设置chunk大小与重叠
3. Chroma本地持久化向量数据库，无需额外部署服务
4. Top‑K语义召回，拿到最相关文档片段
5. Prompt约束大模型，杜绝幻觉，无资料如实回复
6. 打印检索片段日志，方便调试检索质量

## 项目总结
掌握RAG完整链路：文档加载→文本切分→向量化存储→向量检索→Prompt组装→LLM生成回答。理解检索增强生成如何解决大模型幻觉，实现私有知识库问答。
