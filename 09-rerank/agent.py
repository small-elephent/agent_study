import json
import os
import re
from datetime import datetime

import numpy as np
from openai import OpenAI

SYSTEM = (
    "问时间调用 get_time。"
    "其它问题先调用 search_notes。"
    "回答必须带上用到的来源编号，例如 [3]。"
    "工具返回没有相关段落时，必须明确说「笔记里没有」，不要用自己的知识补充。"
)

FOLDER = os.path.dirname(__file__)
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

chunks = None
embed_model = None
vecs = None
rerank_model = None
TOP_K = 3
VEC_FLOOR = 0.40
RERANK_KEEP = 2


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_note():
    path = os.path.join(FOLDER, "knowledge.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def keyword_hit(query, chunk):
    q = (query or "").strip()
    if not q:
        return False
    if q in chunk:
        return True
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", q):
        if word in chunk:
            return True
    hans = "".join(ch for ch in q if "\u4e00" <= ch <= "\u9fff")
    if len(hans) < 2:
        return False
    return any(hans[i : i + 2] in chunk for i in range(len(hans) - 1))


def ensure_search():
    global chunks, embed_model, vecs
    if chunks is not None:
        return
    from modelscope import snapshot_download
    from sentence_transformers import SentenceTransformer

    chunks = split_note()
    embed_model = SentenceTransformer(snapshot_download("BAAI/bge-small-zh-v1.5"))
    vecs = embed_model.encode(chunks)
    print("向量", vecs.shape, flush=True)


def ensure_rerank():
    global rerank_model
    if rerank_model is not None:
        return
    from modelscope import snapshot_download
    from sentence_transformers import CrossEncoder

    rerank_model = CrossEncoder(snapshot_download("BAAI/bge-reranker-base"))
    print("重排模型已加载", flush=True)


def rerank_rows(query, chosen):
    if len(chosen) <= RERANK_KEEP:
        return chosen
    ensure_rerank()
    pairs = [(query, r["chunk"]) for r in chosen]
    scores = rerank_model.predict(pairs)
    for r, s in zip(chosen, scores):
        r["rerank"] = float(s)
    return sorted(chosen, key=lambda r: r["rerank"], reverse=True)[:RERANK_KEEP]


def search_notes(query):
    ensure_search()
    q_vec = embed_model.encode(query)
    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, vecs), 1):
        rows.append(
            {
                "i": i,
                "chunk": chunk,
                "sim": cosine(q_vec, vec),
                "kw": keyword_hit(query, chunk),
            }
        )

    by_sim = sorted(rows, key=lambda r: r["sim"], reverse=True)
    vec_pick = [r for r in by_sim if r["sim"] >= VEC_FLOOR][:TOP_K]
    kw_pick = [r for r in rows if r["kw"]]
    keep_i = {r["i"] for r in vec_pick} | {r["i"] for r in kw_pick}
    chosen = [r for r in by_sim if r["i"] in keep_i]
    if not chosen:
        return "没有相关段落"

    chosen = rerank_rows(query, chosen)
    vec_ids = {r["i"] for r in vec_pick}
    lines = []
    for r in chosen:
        tag = []
        if r["kw"]:
            tag.append("关键词")
        if r["i"] in vec_ids:
            tag.append(f"向量Top{TOP_K}:{r['sim']:.2f}")
        else:
            tag.append(f"向量{r['sim']:.2f}")
        if r.get("rerank") is not None:
            tag.append(f"重排{r['rerank']:.2f}")
        lines.append(f"[{r['i']}] {'+'.join(tag)}\n{r['chunk']}")
    return "\n\n".join(lines)


def cited_ids(text):
    return {int(n) for n in re.findall(r"\[(\d+)\]", text or "")}


def allowed_ids(search_result):
    return {int(n) for n in re.findall(r"^\[(\d+)\]", search_result or "", re.M)}


def looks_like_refuse(answer):
    text = answer or ""
    keys = ("笔记里没有", "没有相关段落", "笔记没有", "资料里没有")
    return any(k in text for k in keys)


def check_grounding(answer, last_search):
    if last_search is None:
        return None
    if last_search == "没有相关段落":
        if looks_like_refuse(answer):
            return None
        return "搜空了却没有明确拒答，可能在用自己的知识"
    used = cited_ids(answer)
    allowed = allowed_ids(last_search)
    if not used:
        return "缺出处：检索有段落，回答里没有 [编号]"
    bad = used - allowed
    if bad:
        return f"假出处：这些编号不在本次检索结果里 {sorted(bad)}"
    return None


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
            "description": "检索笔记段落。除问时间外先调用。返回带编号的段落；没有相关段落时不要编，必须说笔记里没有。",
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
    last_search = None
    for _ in range(5):
        msg = ask(messages)
        messages.append(msg)
        if not msg.tool_calls:
            print(msg.content or "")
            warn = check_grounding(msg.content, last_search)
            if warn:
                print("核对>", warn)
            break
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            print("调用>", tc.function.name)
            print("参数>", tc.function.arguments)
            print("结果>", result)
            if tc.function.name == "search_notes":
                last_search = result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
