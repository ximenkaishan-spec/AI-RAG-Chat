import json
import urllib.request


# Ollama Chat API 地址
url = "http://127.0.0.1:11434/api/chat"


# --------------------------------
# 保存整个聊天历史
# --------------------------------
messages = []


# --------------------------------
# 开始循环，让用户可以连续聊天
# --------------------------------
while True:

    # 获取用户输入
    prompt = input("\n你：")

    # 输入 exit 退出程序
    if prompt.lower() == "exit":
        print("程序结束。")
        break


    # --------------------------------
    # 把用户的问题加入聊天历史
    # --------------------------------
    messages.append({
        "role": "user",
        "content": prompt
    })


    # --------------------------------
    # 准备发送给 Ollama 的数据
    # --------------------------------
    data = {
        "model": "qwen2.5:0.5b",

        # 把完整聊天历史发送给模型
        "messages": messages,

        # false = 一次性返回完整回答
        "stream": False
    }


    # Python 字典 → JSON
    json_data = json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")


    # --------------------------------
    # 创建 HTTP POST 请求
    # --------------------------------
    request = urllib.request.Request(
        url,
        data=json_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )


    # --------------------------------
    # 发送请求
    # --------------------------------
    with urllib.request.urlopen(request) as response:

        # 读取 Ollama 返回的数据
        raw_data = response.read()

        # 二进制 → 字符串
        text_data = raw_data.decode("utf-8")

        # JSON → Python 字典
        result = json.loads(text_data)


    # --------------------------------
    # 从返回结果中提取 AI 回答
    # --------------------------------
    answer = result["message"]["content"]


    # 显示 AI 回答
    print("\nAI：")
    print(answer)


    # --------------------------------
    # 非常重要：
    # 把 AI 的回答也加入聊天历史
    # --------------------------------
    messages.append({
        "role": "assistant",
        "content": answer
    })
