import json
import os
from datetime import datetime
from openai import OpenAI

SYSTEM = (
    "问时间调用 get_time。"
    "问笔记、购物、晚饭时调用 read_note，name 填 memo.txt。"
    "不要编造笔记里没有的内容。"
)

FOLDER = os.path.dirname(__file__)

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_note(name):
    path = os.path.join(FOLDER, name)
    if not os.path.exists(path):
        return "文件不存在"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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
            "name": "read_note",
            "description": "读取笔记。问购物、晚饭、笔记内容时调用。name 填 memo.txt",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
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
    if name == "read_note":
        return read_note(args["name"])
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
            #将结果全部添加到信息中，从而使模型能够记住上下文
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
