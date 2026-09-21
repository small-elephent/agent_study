import json
import os
from datetime import datetime
from openai import OpenAI

SYSTEM = (
    "问时间调用 get_time。"
    "其它问题先调用 search_notes。"
    "工具返回没有相关段落时，就说笔记里没有，不要用自己的知识补充。"
)

FOLDER = os.path.dirname(__file__)

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_note():
    path = os.path.join(FOLDER, "knowledge.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def search_notes(query):
    q = (query or "").strip()
    if not q:
        return "没有相关段落"
    hits = [c for c in split_note() if q in c]
    if not hits and len(q) >= 2:
        hits = [
            c
            for c in split_note()
            if any(q[i : i + 2] in c for i in range(len(q) - 1))
        ]
    return "\n".join(hits) if hits else "没有相关段落"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "问现在几点、今天几号时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "按关键词搜 knowledge.txt 的段落。除了问时间以外先调用。工具返回没有相关段落时，就说笔记没有，不要编。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def load_args(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_tool(name, raw):
    if name == "get_time":
        return get_time()
    args = load_args(raw)
    if args is None:
        return "参数不是合法JSON"
    if name == "search_notes":
        return search_notes(args["query"])
    return f"未知函数 {name}"


def ask(messages):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return resp.choices[0].message


messages = [{"role": "system", "content": SYSTEM}]
while True:
    text = input("你> ").strip()
    if text in ["", "q"]:
        break
    messages.append({"role": "user", "content": text})
    for _ in range(5):
        msg = ask(messages)
        messages.append(msg)
        if not msg.tool_calls:
            print(msg.content or "")
            break
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            print("调用>", tc.function.name)
            print("参数>", tc.function.arguments)
            print("结果>", result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
