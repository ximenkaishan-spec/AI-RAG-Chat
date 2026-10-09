import os
import uuid
import sqlite3
import asyncio
import json
import math
from pathlib import Path

import httpx

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel


# ============================================================
# 1. FastAPI 基础配置
# ============================================================

app = FastAPI(title="AI 私有化知识助手")

app.mount(
    "/static",
    StaticFiles(directory="/app/static"),
    name="static"
)


# ============================================================
# 2. Ollama / Embedding / Qwen 配置
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.docker.internal:11434"
)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "nomic-embed-text"
)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MODEL_NAME = os.getenv(
    "DASHSCOPE_MODEL",
    "qwen-plus-2025-07-28"
)


# ============================================================
# 3. 数据目录
# ============================================================

BASE_DIR = Path("/app")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "chat.db"

DOCUMENT_DIR = DATA_DIR / "documents"
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. RAG 参数
# ============================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD = float(
    os.getenv("RAG_SIMILARITY_THRESHOLD", "0.63")
)


# ============================================================
# 5. 正在生成的任务
# ============================================================

GENERATION_TASKS = {}


# ============================================================
# 6. SQLite 初始化
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_name TEXT PRIMARY KEY,
            stored_filename TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# 7. 首页
# ============================================================

@app.get("/")
async def index():
    return FileResponse("/app/static/index.html")


# ============================================================
# 8. 数据模型
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


# ============================================================
# 9. 保存聊天消息
# ============================================================

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    conn.commit()
    conn.close()


# ============================================================
# 10. 获取会话历史
# ============================================================

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, created_at
        FROM conversations
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()

    messages = []
    for role, content, created_at in rows:
        messages.append({
            "role": role,
            "content": content,
            "created_at": created_at
        })

    return {"session_id": session_id, "messages": messages}


# ============================================================
# 11. 获取会话列表
# ============================================================

@app.get("/sessions")
async def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT session_id, MIN(created_at) AS created_at
        FROM conversations
        GROUP BY session_id
        ORDER BY MIN(id) DESC
    """)
    rows = cursor.fetchall()

    sessions = []
    for session_id, created_at in rows:
        cursor.execute("""
            SELECT content
            FROM conversations
            WHERE session_id = ? AND role = 'user'
            ORDER BY id ASC LIMIT 1
        """, (session_id,))
        first_message = cursor.fetchone()
        title = first_message[0] if first_message else "新会话"
        if len(title) > 40:
            title = title[:40] + "..."
        sessions.append({
            "session_id": session_id,
            "title": title,
            "created_at": created_at
        })

    conn.close()
    return {"sessions": sessions}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(400, "session_id 不能为空")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM conversations WHERE session_id = ?",
        (session_id,)
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    return {"ok": True, "deleted": deleted, "session_id": session_id}


# ============================================================
# 12. 文本切分
# ============================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
):
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = end - chunk_overlap

    return chunks


# ============================================================
# 13. Embedding
# ============================================================

async def get_embedding(text: str):
    payload = {"model": EMBED_MODEL, "prompt": text}

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError("Ollama 没有返回 embedding")
    return embedding


# ============================================================
# 14. 余弦相似度
# ============================================================

def cosine_similarity(vector_a, vector_b):
    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"向量维度不一致：{len(vector_a)} != {len(vector_b)}"
        )

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# 15. 预检：文档是否已存在
# ============================================================

@app.get("/documents/exists/{document_name:path}")
async def document_exists(document_name: str):
    document_name = document_name.strip()
    if not document_name:
        raise HTTPException(400, "document_name 不能为空")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM documents WHERE document_name = ?",
        (document_name,)
    )
    row = cursor.fetchone()
    conn.close()

    return {"document_name": document_name, "exists": bool(row)}


# ============================================================
# 16. 上传知识库文档
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
):
    original_name = file.filename or ""
    if not original_name:
        raise HTTPException(400, "文件名不能为空")

    suffix = Path(original_name).suffix.lower()
    if suffix not in {".txt", ".md", ".markdown"}:
        raise HTTPException(400, "仅支持 .txt、.md、.markdown 文件")

    content_bytes = await file.read()
    if len(content_bytes) > 5 * 1024 * 1024:
        raise HTTPException(400, "文件不能超过 5MB")

    try:
        text_content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "文件必须使用 UTF-8 编码")

    chunks = split_text(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(400, "文件内容为空")

    stored_filename = f"{uuid.uuid4().hex}{suffix}"
    stored_path = DOCUMENT_DIR / stored_filename
    stored_path.write_bytes(content_bytes)

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stored_filename FROM documents
            WHERE document_name = ?
        """, (original_name,))
        old_row = cursor.fetchone()
        old_stored_filename = old_row[0] if old_row else None

        if old_row and not overwrite:
            if stored_path.exists():
                try:
                    stored_path.unlink()
                except Exception:
                    pass
            raise HTTPException(
                409,
                f"文档「{original_name}」已存在，未选择覆盖。"
            )

        # 生成 Embedding（耗时最长的步骤，前端进度条在此阶段）
        embeddings = []
        for chunk in chunks:
            embedding = await get_embedding(chunk)
            if not embedding:
                raise RuntimeError(
                    f"Embedding 生成失败，Chunk index={len(embeddings)}"
                )
            embeddings.append(embedding)

        # 旧文件重命名为 .old 备份
        old_file_backup = None
        if old_stored_filename and old_stored_filename != stored_filename:
            old_path = DOCUMENT_DIR / old_stored_filename
            if old_path.exists():
                old_file_backup = old_path.with_suffix(
                    old_path.suffix + ".old"
                )
                if old_file_backup.exists():
                    try:
                        old_file_backup.unlink()
                    except Exception:
                        pass
                old_path.rename(old_file_backup)

        cursor.execute("BEGIN")

        cursor.execute("""
            DELETE FROM rag_chunks WHERE document_name = ?
        """, (original_name,))

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cursor.execute("""
                INSERT INTO rag_chunks (
                    document_name, chunk_index, content, embedding
                ) VALUES (?, ?, ?, ?)
            """, (
                original_name, index, chunk, json.dumps(embedding)
            ))

        cursor.execute("""
            INSERT INTO documents (document_name, stored_filename)
            VALUES (?, ?)
            ON CONFLICT(document_name)
            DO UPDATE SET stored_filename = excluded.stored_filename
        """, (original_name, stored_filename))

        conn.commit()

        old_file_deleted = True
        old_file_delete_error = None

        if old_file_backup and old_file_backup.exists():
            try:
                old_file_backup.unlink()
            except Exception as e:
                old_file_deleted = False
                old_file_delete_error = str(e)

        result = {
            "success": True,
            "document_name": original_name,
            "chunks": len(chunks),
            "chunk_count": len(chunks),
            "stored_filename": stored_filename,
            "overwritten": bool(old_row),
            "old_file_deleted": old_file_deleted
        }
        if old_file_delete_error:
            result["warning"] = (
                "新文档已保存成功，但旧源文件删除失败："
                + old_file_delete_error
            )
        return result

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        try:
            if 'old_file_backup' in locals() and old_file_backup and old_file_backup.exists():
                restored = DOCUMENT_DIR / old_stored_filename
                if not restored.exists():
                    old_file_backup.rename(restored)
        except Exception:
            pass

        if stored_path.exists():
            try:
                stored_path.unlink()
            except Exception:
                pass

        raise HTTPException(500, f"文档处理失败：{str(e)}")

    finally:
        conn.close()


# ============================================================
# 17. 文档列表
# ============================================================

@app.get("/documents")
async def list_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            document_name,
            COUNT(*) AS chunk_count,
            MIN(created_at) AS created_at
        FROM rag_chunks
        GROUP BY document_name
        ORDER BY MIN(id) DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    documents = []
    for document_name, chunk_count, created_at in rows:
        documents.append({
            "document_name": document_name,
            "chunk_count": chunk_count,
            "created_at": created_at
        })

    return {"documents": documents}


# ============================================================
# 18. 查看文档源文件内容
#
# 新增：is_markdown 字段，前端据此决定默认渲染模式
# ============================================================

@app.get("/documents/content/{document_name:path}")
async def get_document_content(document_name: str):
    document_name = document_name.strip()
    if not document_name:
        raise HTTPException(400, "document_name 不能为空")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stored_filename FROM documents
            WHERE document_name = ?
        """, (document_name,))
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(404, "文档不存在")

    stored_filename = row[0]
    stored_path = DOCUMENT_DIR / stored_filename

    if not stored_path.exists():
        raise HTTPException(404, "源文件已丢失")

    try:
        content = stored_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = stored_path.read_bytes().decode(
            "utf-8", errors="replace"
        )
    except Exception as e:
        raise HTTPException(500, f"读取源文件失败：{e}")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM rag_chunks
            WHERE document_name = ?
        """, (document_name,))
        chunk_count = cursor.fetchone()[0]
    finally:
        conn.close()

    is_markdown = Path(stored_filename).suffix.lower() in {
        ".md", ".markdown"
    }

    return {
        "document_name": document_name,
        "content": content,
        "source": "file",
        "chunk_count": chunk_count,
        "is_markdown": is_markdown
    }


# ============================================================
# 19. 删除文档
# ============================================================

@app.delete("/documents/{document_name:path}")
async def delete_document(document_name: str):
    document_name = document_name.strip()
    if not document_name:
        raise HTTPException(400, "document_name 不能为空")

    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT stored_filename FROM documents
            WHERE document_name = ?
        """, (document_name,))
        document_row = cursor.fetchone()
        stored_filename = document_row[0] if document_row else None

        cursor.execute("""
            SELECT COUNT(*) FROM rag_chunks
            WHERE document_name = ?
        """, (document_name,))
        chunk_count = cursor.fetchone()[0]

        if not document_row and chunk_count == 0:
            raise HTTPException(404, "文档不存在")

        deleting_path = None
        original_path = None

        if stored_filename:
            original_path = DOCUMENT_DIR / stored_filename
            if original_path.exists():
                deleting_path = original_path.with_suffix(
                    original_path.suffix + ".deleting"
                )
                if deleting_path.exists():
                    try:
                        deleting_path.unlink()
                    except Exception:
                        pass
                original_path.rename(deleting_path)

        cursor.execute("BEGIN")

        cursor.execute("""
            DELETE FROM rag_chunks WHERE document_name = ?
        """, (document_name,))
        deleted_chunks = cursor.rowcount

        cursor.execute("""
            DELETE FROM documents WHERE document_name = ?
        """, (document_name,))

        conn.commit()

        source_deleted = True
        source_delete_error = None

        if deleting_path and deleting_path.exists():
            try:
                deleting_path.unlink()
            except Exception as e:
                source_deleted = False
                source_delete_error = str(e)

        result = {
            "success": True,
            "document_name": document_name,
            "deleted_chunks": deleted_chunks,
            "source_deleted": source_deleted
        }
        if source_delete_error:
            result["warning"] = (
                "数据库数据已删除，但源文件删除失败："
                + source_delete_error
            )
        return result

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        try:
            if 'deleting_path' in locals() and deleting_path and deleting_path.exists() and original_path:
                if not original_path.exists():
                    deleting_path.rename(original_path)
        except Exception:
            pass
        raise HTTPException(500, f"删除文档失败：{str(e)}")

    finally:
        conn.close()


# ============================================================
# 20. Query Rewrite
# ============================================================

async def rewrite_query(user_message: str, history_rows):
    current = user_message.strip()
    if not current:
        return current

    history_for_rewrite = []
    current_skipped = False

    for role, content in reversed(history_rows):
        content = content.strip()
        if not content:
            continue

        if role == "user":
            if not current_skipped and content == current:
                current_skipped = True
                continue
            history_for_rewrite.append(f"用户：{content}")
        elif role == "assistant":
            history_for_rewrite.append(f"助手：{content}")

        if len(history_for_rewrite) >= 6:
            break

    history_for_rewrite.reverse()
    history_text = "\n".join(history_for_rewrite)

    if not history_text:
        return current

    system_prompt = """
你是一个 RAG 检索 Query Rewrite 助手。

你的唯一任务是：
根据最近的对话历史，把用户当前问题改写成一个
语义完整、明确、适合知识库向量检索的问题。

规则：
1. 如果当前问题已经完整明确，直接原样返回。
2. 如果当前问题包含“它、这个、那个、该技术、该工具、这个方案”等指代词，
   根据对话历史补充明确的实体。
3. 不要回答用户的问题。
4. 不要解释改写过程。
5. 不要添加知识库中不存在的信息。
6. 只输出最终的检索 Query。
7. 尽量保持用户原问题的意思不变。
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": (
                "最近对话：\n"
                f"{history_text}\n\n"
                "当前问题：\n"
                f"{current}\n\n"
                "请输出改写后的检索 Query。"
            )
        }
    ]

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "temperature": 0
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{DASHSCOPE_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            rewritten_query = (
                data["choices"][0]["message"]["content"].strip()
            )
            if not rewritten_query:
                return current

            print("========== Query Rewrite ==========")
            print(f"原始问题：{current}")
            print(f"改写结果：{rewritten_query}")
            print("====================================")
            return rewritten_query

    except Exception as e:
        print("========== Query Rewrite 失败 ==========")
        print(f"错误：{e}")
        print("将使用原始问题继续。")
        print("========================================")
        return current


# ============================================================
# 21. RAG 检索
# ============================================================

async def retrieve_documents(
    question: str,
    top_k: int = RAG_TOP_K,
    threshold: float = RAG_SIMILARITY_THRESHOLD
):
    question_vector = await get_embedding(question)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, document_name, chunk_index, content, embedding
        FROM rag_chunks
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    results = []
    for row in rows:
        embedding = json.loads(row[4])
        similarity = cosine_similarity(question_vector, embedding)
        if similarity >= threshold:
            results.append({
                "id": row[0],
                "document_name": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "similarity": similarity
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ============================================================
# 22. RAG 调试
# ============================================================

@app.get("/rag/debug")
async def rag_debug(question: str):
    question = question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")

    question_vector = await get_embedding(question)
    embedding_dimension = len(question_vector)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, document_name, chunk_index, content, embedding
        FROM rag_chunks
    """)
    rows = cursor.fetchall()
    conn.close()

    candidates = []
    for row in rows:
        embedding = json.loads(row[4])
        similarity = cosine_similarity(question_vector, embedding)
        candidates.append({
            "id": row[0],
            "document_name": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "similarity": similarity
        })

    passed_results = [
        item for item in candidates
        if item["similarity"] >= RAG_SIMILARITY_THRESHOLD
    ]
    passed_results.sort(
        key=lambda item: item["similarity"], reverse=True
    )
    results = passed_results[:RAG_TOP_K]

    max_similarity = max(
        (item["similarity"] for item in candidates),
        default=0.0
    )
    gate_pass = max_similarity >= RAG_SIMILARITY_THRESHOLD

    return {
        "question": question,
        "embedding_dimension": embedding_dimension,
        "max_similarity": max_similarity,
        "threshold": RAG_SIMILARITY_THRESHOLD,
        "gate_pass": gate_pass,
        "top_k": RAG_TOP_K,
        "candidate_count": len(candidates),
        "threshold_pass_count": len(passed_results),
        "result_count": len(results),
        "results": results
    }


# ============================================================
# 23. RAG Prompt
# ============================================================

def build_rag_prompt(question: str, retrieved_chunks):
    context_parts = []
    for index, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[知识片段 {index}]\n"
            f"文档：{item['document_name']}\n"
            f"内容：{item['content']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
你是一个知识库问答助手。

请根据下面提供的【知识库内容】回答用户问题。

====================
回答规则
====================

1. 只能根据【知识库内容】回答。
2. 不要使用知识库之外的预训练知识、常识或网络信息补充答案。
3. 如果多个知识片段都与问题相关，可以综合这些知识片段进行回答。
4. 可以对知识库内容进行归纳、总结和简单解释。
5. 回答必须有知识库内容作为依据。
6. 不要因为知识库中的原文没有直接出现用户问题的完整问法，就拒绝回答。
7. 如果知识库能够支持回答，直接给出答案，不要讨论“是否能够回答”。
8. 回答简洁、直接、准确。
9. 不要解释内部推理过程。

====================
知识库内容
====================

{context}

====================
用户问题
====================

{question}

====================
请直接回答
====================
"""
    return prompt


# ============================================================
# 24. Qwen 流式生成
# ============================================================

async def qwen_stream(session_id: str, messages: list):
    assistant_content = ""
    current_task = asyncio.current_task()

    GENERATION_TASKS[session_id] = current_task

    print(f"========== Qwen 生成任务开始：{session_id} ==========")

    request_data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            print("========== 准备调用百炼 HTTP API ==========")

            async with client.stream(
                "POST",
                f"{DASHSCOPE_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=request_data
            ) as response:

                print(
                    f"========== 百炼 HTTP 状态码："
                    f"{response.status_code} =========="
                )

                if response.status_code >= 400:
                    error_body = await response.aread()
                    print("========== 百炼错误响应 ==========")
                    print(error_body.decode("utf-8", errors="ignore"))
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    print(f"百炼原始 SSE：{line}")

                    if not line.startswith("data:"):
                        continue

                    data_text = line[len("data:"):].strip()

                    if data_text == "[DONE]":
                        break

                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError as e:
                        print(f"百炼 SSE JSON 解析失败：{e}")
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if isinstance(content, str):
                        if content:
                            assistant_content += content
                            yield content
                    elif isinstance(content, list):
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            text = item.get("text", "")
                            if text:
                                assistant_content += text
                                yield text

    except asyncio.CancelledError:
        print(
            f"========== Qwen 生成任务被取消：{session_id} =========="
        )
        if assistant_content.strip():
            save_message(session_id, "assistant", assistant_content)
        raise

    except Exception as e:
        print(f"========== 百炼请求失败：{e} ==========")
        yield "\n\n[AI 服务请求失败，请检查百炼配置。]"

    else:
        if assistant_content.strip():
            save_message(session_id, "assistant", assistant_content)
        print(f"========== Qwen 生成完成：{session_id} ==========")

    finally:
        if GENERATION_TASKS.get(session_id) is current_task:
            GENERATION_TASKS.pop(session_id, None)
        print(f"========== Qwen 生成任务清理：{session_id} ==========")


# ============================================================
# 25. AI 对话接口
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id.strip()
    user_message = request.message.strip()

    if not session_id:
        raise HTTPException(400, "session_id 不能为空")
    if not user_message:
        raise HTTPException(400, "message 不能为空")

    existing_task = GENERATION_TASKS.get(session_id)
    if existing_task and not existing_task.done():
        raise HTTPException(409, "当前会话已有正在生成的任务")

    save_message(session_id, "user", user_message)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content
        FROM conversations
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))
    history_rows = cursor.fetchall()
    conn.close()

    rewritten_query = await rewrite_query(user_message, history_rows)

    print("========== RAG 检索问题 ==========")
    print(f"原始问题：{user_message}")
    print(f"Rewrite 后检索问题：{rewritten_query}")
    print("==================================")

    try:
        retrieved_chunks = await retrieve_documents(rewritten_query)
    except Exception as e:
        print(f"RAG 检索失败：{e}")
        error_text = (
            "RAG 检索服务异常，请检查 Embedding / Ollama 服务。"
        )

        async def rag_error_generator():
            yield error_text

        return StreamingResponse(
            rag_error_generator(),
            media_type="text/plain; charset=utf-8"
        )

    print("============================================")
    print(f"/chat 实际 retrieved_chunks 数量：{len(retrieved_chunks)}")
    if retrieved_chunks:
        print(
            f"/chat 实际最高相似度："
            f"{retrieved_chunks[0]['similarity']:.4f}"
        )
    else:
        print("/chat 实际没有检索到任何 Chunk")

    print("========== RAG 闸门 ==========")
    print(
        f"最高相似度："
        f"{retrieved_chunks[0]['similarity']:.4f}"
        if retrieved_chunks else "最高相似度：0.0000"
    )
    print(f"最低通过阈值：{RAG_SIMILARITY_THRESHOLD:.4f}")

    if retrieved_chunks:
        print("✓ 最高相似度达到阈值")
        print("✓ 允许进入百炼大语言模型")
    else:
        print("✕ 最高相似度没有达到阈值")
        print("✕ 拒绝进入百炼大语言模型")

    print("============================================")

    if not retrieved_chunks:
        refusal_text = "知识库中没有找到相关信息。"

        async def refusal_generator():
            save_message(session_id, "assistant", refusal_text)
            yield refusal_text

        return StreamingResponse(
            refusal_generator(),
            media_type="text/plain; charset=utf-8"
        )

    rag_prompt = build_rag_prompt(user_message, retrieved_chunks)

    messages = [{"role": "system", "content": rag_prompt}]
    for role, content in history_rows:
        messages.append({"role": role, "content": content})

    if not DASHSCOPE_API_KEY:
        raise HTTPException(500, "DASHSCOPE_API_KEY 未配置")

    print("========== 发送给百炼的 messages ==========")
    for index, message in enumerate(messages, start=1):
        print(f"【消息 {index}】")
        print(f"role: {message['role']}")
        print(message["content"])
        print("--------------------------------------------")
    print("========== 已完成百炼请求准备 ==========")

    return StreamingResponse(
        qwen_stream(session_id, messages),
        media_type="text/plain; charset=utf-8"
    )


# ============================================================
# 26. 停止生成
# ============================================================

@app.post("/stop/{session_id}")
async def stop_generation(session_id: str):
    task = GENERATION_TASKS.get(session_id)

    if task is None:
        return {
            "message": "当前没有正在生成的任务",
            "session_id": session_id,
            "stopped": False
        }

    if task.done():
        GENERATION_TASKS.pop(session_id, None)
        return {
            "message": "生成任务已经结束",
            "session_id": session_id,
            "stopped": False
        }

    print(f"========== 收到停止请求：{session_id} ==========")
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print(
            f"========== 生成任务取消完成：{session_id} =========="
        )

    return {
        "message": "生成任务已取消",
        "session_id": session_id,
        "stopped": True
    }


# ============================================================
# 27. 健康检查
# ============================================================

@app.get("/health")
async def health():
    checks = {}

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conversation_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM rag_chunks")
        rag_chunk_count = cursor.fetchone()[0]
        conn.close()

        checks["sqlite"] = {
            "status": "ok",
            "message": "SQLite 正常",
            "conversation_count": conversation_count,
            "rag_chunk_count": rag_chunk_count
        }
    except Exception as e:
        checks["sqlite"] = {"status": "error", "message": str(e)}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
        models = data.get("models", [])
        checks["ollama"] = {
            "status": "ok",
            "message": "Ollama 正常",
            "model_count": len(models)
        }
    except Exception as e:
        checks["ollama"] = {"status": "error", "message": str(e)}

    try:
        vector = await get_embedding("health check")
        checks["embedding"] = {
            "status": "ok",
            "message": "Embedding 正常",
            "model": EMBED_MODEL,
            "dimension": len(vector)
        }
    except Exception as e:
        checks["embedding"] = {"status": "error", "message": str(e)}

    if not DASHSCOPE_API_KEY:
        checks["qwen"] = {
            "status": "error",
            "message": "DASHSCOPE_API_KEY 未配置"
        }
    else:
        try:
            test_messages = [
                {"role": "user", "content": "请只回答：OK"}
            ]
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": MODEL_NAME,
                        "messages": test_messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
            checks["qwen"] = {
                "status": "ok",
                "message": "Qwen / 百炼正常",
                "model": MODEL_NAME
            }
        except Exception as e:
            checks["qwen"] = {"status": "error", "message": str(e)}

    overall_ok = all(
        item.get("status") == "ok"
        for item in checks.values()
    )

    return {
        "status": "ok" if overall_ok else "error",
        "checks": checks
    }