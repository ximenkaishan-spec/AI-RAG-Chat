# AI 私有化知识库问答系统（ AI RAG Chat）

基于 FastAPI + Docker + Ollama + Embedding + RAG + 通义千问构建的 AI 知识库问答 Demo。

项目重点模拟企业内部 AI 私有化部署、知识库问答、系统集成与运维场景。

---

## 1. 项目简介

本项目实现了一个可部署的 AI 知识库问答系统。

用户可以：

- 创建会话
- 进行多轮 AI 对话
- 上传知识库文档
- 对知识库进行语义检索
- 基于检索结果生成回答
- 查看历史会话
- 删除知识库文档
- 实时流式接收 AI 输出
- 手动停止正在生成的回答
- 查看系统健康状态
- 调试 RAG 检索结果

项目重点不是单纯调用大模型 API，而是完整实现：

用户
→ Web 前端
→ FastAPI
→ RAG 检索
→ Ollama Embedding
→ 百炼 Qwen
→ 流式返回

这一套完整链路。

---

## 2. 技术栈

### 后端

- Python
- FastAPI
- Uvicorn
- SQLite
- HTTPX

### AI

- Ollama
- nomic-embed-text
- 通义千问 Qwen Plus
- RAG
- Cosine Similarity

### 前端

- HTML
- JavaScript
- marked.js
- DOMPurify
- Highlight.js

### 部署

- Docker
- Ubuntu 22.04
- 阿里云轻量应用服务器

---

## 3. 系统架构

```text
┌──────────────────────────────┐
│          Web Browser         │
│  HTML / JavaScript / Markdown│
└──────────────┬───────────────┘
               │ HTTP
               ▼
┌──────────────────────────────┐
│          FastAPI             │
│                              │
│  /chat                       │
│  /documents                  │
│  /documents/upload           │
│  /rag/debug                  │
│  /health                     │
│  /stop/{session_id}          │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌─────────────────┐
│   SQLite     │  │ Ollama           │
│              │  │ Embedding        │
│ 会话历史     │  │ nomic-embed-text │
│ RAG 文档     │  └────────┬────────┘
└──────────────┘           │
                           │ 向量
                           ▼
                    ┌──────────────┐
                    │ RAG 检索     │
                    │ Top-K = 3    │
                    │ 阈值 = 0.63  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ 百炼 Qwen    │
                    │ Qwen Plus    │
                    └──────────────┘
## 4. RAG 工作流程
文档上传
上传 TXT / Markdown
        ↓
文本解析
        ↓
文档切块
        ↓
Embedding
        ↓
保存 SQLite

当前默认参数：

CHUNK_SIZE=500
CHUNK_OVERLAP=100
用户提问
用户问题
   ↓
问题 Embedding
   ↓
Cosine Similarity
   ↓
按照相似度排序
   ↓
取 Top-K
   ↓
相似度闸门
   ↓
通过 → 构建 RAG Prompt
   ↓
调用 Qwen
   ↓
流式返回

如果最高相似度低于：

RAG_SIMILARITY_THRESHOLD=0.63

系统不会调用大模型，而是直接返回：

知识库中没有找到相关信息。

这样可以降低模型脱离知识库自行回答的风险。

## 5. 当前 RAG 配置
配置	当前值
Embedding	nomic-embed-text
Chunk Size	500
Chunk Overlap	100
Top-K	3
Similarity Threshold	0.63
Vector Dimension	768
## 6. API
健康检查
GET /health

检查：

SQLite
Ollama
Embedding
百炼 API
AI 对话
POST /chat

支持：

RAG
多轮历史
流式输出
服务端停止生成
停止生成
POST /stop/{session_id}

服务端通过 asyncio Task 管理正在执行的百炼请求。

核心流程：

用户点击停止
      ↓
POST /stop/{session_id}
      ↓
找到后台生成 Task
      ↓
task.cancel()
      ↓
关闭百炼流式请求
      ↓
保存已经生成的内容
      ↓
清理 Task
知识库列表
GET /documents

返回当前知识库中的文档及 Chunk 数量。

删除知识库文档
DELETE /documents/{document_name}

删除指定文档对应的全部 RAG Chunk。

RAG 调试
GET /rag/debug?question=你的问题

用于查看：

Embedding 维度
Top-K
相似度阈值
最高相似度
通过闸门的 Chunk 数量
实际召回内容
文档上传
POST /documents/upload

支持：

.txt
.md
## 7. 环境变量

项目使用 .env 管理运行配置。

示例：

DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus-2025-07-28

EMBED_MODEL=nomic-embed-text

CHUNK_SIZE=500
CHUNK_OVERLAP=100
RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.63

真实 API Key 不应该提交到 Git。

项目提供：

.env.example

作为配置模板。

## 8. Docker 部署
构建镜像
docker build -t ai-demo .
创建容器
docker run -d \
  --name ai-demo \
  --env-file ~/ai-demo/.env \
  --add-host=host.docker.internal:host-gateway \
  -p 8000:8000 \
  -v /root/ai-demo/data:/app/data \
  ai-demo
查看容器
docker ps
查看日志
docker logs ai-demo
查看最近日志
docker logs --tail 100 ai-demo
实时日志
docker logs -f ai-demo
## 9. 数据持久化

项目使用 Docker Bind Mount：

/root/ai-demo/data
        ↓
/app/data

SQLite 数据及知识库数据保存在：

data/

因此重新创建 Docker 容器后，业务数据不会因为容器删除而丢失。

## 10. 项目目录
ai-demo/
├── api.py
├── app.py
├── Dockerfile
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── chat.db
│   └── documents/
└── static/
    ├── index.html
    ├── marked.min.js
    ├── purify.min.js
    ├── highlight.min.js
    └── highlight.css
## 11. 安全注意事项
API Key

不要把真实 API Key 写入：

Git
GitHub
Gitee
README
Dockerfile
前端 JavaScript

真实配置只放在：

.env

并建议：

chmod 600 .env
## 12. 已实现功能
 FastAPI
 Docker 部署
 Ollama Embedding
 Qwen Plus
 RAG
 文档切块
 Cosine Similarity
 RAG 相似度闸门
 Top-K 检索
 RAG Debug
 SQLite 会话历史
 多轮对话
 知识库上传
 知识库列表
 知识库删除
 Markdown
 DOMPurify XSS 防护
 Highlight.js 代码高亮
 流式输出
 浏览器停止生成
 服务端停止生成
 健康检查
 环境变量配置
 Docker 数据持久化
## 13. 项目定位

本项目用于展示以下能力：

Linux
Docker
Python
FastAPI
AI 应用部署
大模型 API 集成
Ollama
Embedding
RAG
SQLite
HTTP API
流式响应
异步任务
系统运维
故障排查

重点体现 AI 私有化部署与系统集成场景中的：

部署
配置
联调
排障
监控
数据持久化
知识库管理
模型调用
服务端任务控制
## 14. 后续可以继续扩展
 知识库 Web 管理界面
 上传进度
 Markdown 文件完整支持
 文档重新索引
 文档预览
 用户认证
 权限管理
 Nginx 反向代理
 HTTPS
 Docker Compose
 Redis
 PostgreSQL
 向量数据库
 Prometheus + Grafana
 Kubernetes 部署

