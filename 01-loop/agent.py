import os
from datetime import datetime
from openai import OpenAI

# 说明书：什么时候该伸手去看钟。这不是 RAG，没有检索。
SYSTEM = "需要当前时间时调用 get_time。不要自己猜时间。"

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 交给模型的工具说明书：名字、干什么、参数。模型不会真的看钟。
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "问现在几点、今天几号时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def run_tool(name, raw):
    if name == "get_time":
        return get_time()
    return f"未知函数 {name}"


def ask(messages):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        extra_body={"thinking": {"type": "enabled"}},#显示思考过程
    )
    return resp.choices[0].message

#函数开始
messages = [{"role": "system", "content": SYSTEM}]
while True:
    text = input("你> ").strip()
    if text in ["", "q"]:
        break
    messages.append({"role": "user", "content": text})
    # 最多 5 圈：调工具 → 看结果 → 再决定。没有 tool_calls 就该开口回答了。
    for _ in range(5):
        msg = ask(messages)
        messages.append(msg)
        if not msg.tool_calls:
            print(msg.content or "")
            break
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            print("调用>", tc.function.name)
            print("结果>", result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
