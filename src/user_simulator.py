"""
用户模拟器 - 用 LLM 扮演不同 persona 的用户
生成多样化对话，供评测系统使用
"""

import json
import os
import socket
import re
import time

EXTRA_HEADERS = {
    "Adams-Platform-User": os.environ.get("ADAMS_PLATFORM_USER", ""),
    "Adams-User-Token": os.environ.get("ADAMS_USER_TOKEN", ""),
    "Adams-Business": os.environ.get("ADAMS_BUSINESS", ""),
}

IPS = None

def get_ips():
    global IPS
    if IPS is None:
        IPS = list(set(a[4][0] for a in socket.getaddrinfo("mmdcadamsminiserverproxy.polaris", 25340, socket.AF_INET, socket.SOCK_STREAM)))
    return IPS

def call_llm(messages, max_tokens=4096, temperature=0.7):
    ips = get_ips()
    payload = json.dumps({
        "model": "kimi_k2d6",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    })
    body_path = "/service/20839/v1/chat/completions"
    http_req = f"POST {body_path} HTTP/1.1\r\nHost: mmdcadamsminiserverproxy.polaris:25340\r\nContent-Type: application/json\r\nConnection: close\r\n"
    for k, v in EXTRA_HEADERS.items():
        http_req += f"{k}: {v}\r\n"
    http_req += f"Content-Length: {len(payload)}\r\n\r\n{payload}"

    for ip in ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(120)
        try:
            s.connect((ip, 25340))
            s.sendall(http_req.encode())
            resp = b""
            while True:
                try:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    resp += chunk
                except socket.timeout:
                    break
            decoded = resp.decode("utf-8", errors="replace")
            parts = decoded.split("\r\n\r\n", 1)
            if len(parts) < 2:
                continue
            body = parts[1]
            if "transfer-encoding: chunked" in parts[0].lower():
                chunks = []
                pos = 0
                while pos < len(body):
                    line_end = body.find("\r\n", pos)
                    if line_end < 0:
                        break
                    size_str = body[pos:line_end].strip()
                    if not size_str:
                        pos = line_end + 2
                        continue
                    try:
                        chunk_size = int(size_str, 16)
                    except ValueError:
                        break
                    if chunk_size == 0:
                        break
                    chunk_start = line_end + 2
                    chunks.append(body[chunk_start:chunk_start + chunk_size])
                    pos = chunk_start + chunk_size + 2
                body = "".join(chunks)
            data = json.loads(body)
            return data["choices"][0]["message"].get("content") or ""
        except Exception:
            continue
        finally:
            s.close()
    raise RuntimeError("All IPs failed")


PERSONAS = {
    "cooperative": {
        "name": "配合型用户",
        "personality": "配合，但会自然地提问几个问题",
        "special": "",
    },
    "reluctant": {
        "name": "犹豫型用户",
        "personality": "不太想配合，会犹豫，但最终可能同意",
        "special": "不太确定自己能不能做到要求",
    },
    "rejecting": {
        "name": "拒绝型用户",
        "personality": "明确不想配合，找各种理由拒绝",
        "special": "表示自己很忙，不想接这个电话",
    },
    "curious": {
        "name": "追问型用户",
        "personality": "对细节很多疑问，反复追问合同条款",
        "special": "会问很多关于合同、退出、奖励的问题",
    },
    "driving": {
        "name": "开车型用户",
        "personality": "正在开车，不方便说话",
        "special": "一开始就说明自己在开车",
    },
}


def simulate_dialogue(instruction: str, persona_type: str = "cooperative", max_turns: int = 12) -> list[dict]:
    """
    用 LLM 模拟用户，和"被测Agent"生成一段完整对话。
    
    注意：这里被测Agent也是LLM扮演的（根据指令），不是真正的被测系统。
    这是为了生成测试数据。后续会替换成真正的被测Agent API。
    """
    persona = PERSONAS.get(persona_type, PERSONAS["cooperative"])

    # 提取指令中的角色和开场白
    role_match = re.search(r'#?\s*Role[:：\n]\s*(.+?)(?:\n#|\n\n|\n[A-Z]|\Z)', instruction, re.DOTALL)
    agent_role = role_match.group(1).strip() if role_match else "客服"

    opening_match = re.search(r'#?\s*Opening Line[:：]\s*(.+?)(?:\n#|\n\n|\Z)', instruction, re.DOTALL)
    opening_line = opening_match.group(1).strip() if opening_match else "您好"
    # 替换变量
    opening_line = re.sub(r'\$\{[^}]+\}', '张三', opening_line)

    # 构建用户模拟器的 system prompt
    user_system = f"""你是一个模拟真实用户的角色。你正在接听一个外呼电话。

【来电方】{agent_role}

【你的人物设定】
- 性格：{persona['personality']}
{"- 特殊情况：" + persona['special'] if persona['special'] else ""}

【行为规则】
1. 像真实用户一样自然回复，用口语化短句
2. 严格按照你的人物设定行事
3. 不要主动推进对话，等对方说
4. 每次回复控制在20字以内
5. 不要用书面语，用日常口语
6. 如果对方说完了你想说的，可以主动结束（说"好的挂了""再见"等）

只输出你的回复内容，不要加引号、不要加角色标记。"""

    # 构建被测Agent的 system prompt
    agent_system = f"""你是一个外呼客服Agent，需要严格按照以下指令执行通话。

{instruction}

请严格按照指令中的Role、Call Flow、Knowledge Points和Constraints执行。
注意字数限制和语气要求。
每次回复后等待用户回应。

只输出你的回复内容，不要加引号、不要加角色标记。"""

    dialogue = []

    # 第一轮：Agent 发起开场白
    dialogue.append({"role": "agent", "content": opening_line, "turn": 1})

    # 交替对话
    user_messages = [{"role": "system", "content": user_system}]
    agent_messages = [{"role": "system", "content": agent_system}]

    for turn in range(2, max_turns + 1):
        # 用户回复
        last_msg = dialogue[-1]["content"]
        if dialogue[-1]["role"] == "agent":
            user_messages.append({"role": "user", "content": f"对方说: {last_msg}\n\n你的回复:"})
            try:
                user_reply = call_llm(user_messages, max_tokens=200, temperature=0.7)
                user_reply = user_reply.strip().strip('"').strip()
            except Exception:
                user_reply = "嗯"
            user_messages.append({"role": "assistant", "content": user_reply})
            dialogue.append({"role": "user", "content": user_reply, "turn": turn})
        else:
            # Agent回复
            agent_messages.append({"role": "user", "content": f"对方说: {dialogue[-1]['content']}\n\n你的回复:"})
            try:
                agent_reply = call_llm(agent_messages, max_tokens=200, temperature=0.5)
                agent_reply = agent_reply.strip().strip('"').strip()
            except Exception:
                agent_reply = "好的"
            agent_messages.append({"role": "assistant", "content": agent_reply})
            dialogue.append({"role": "agent", "content": agent_reply, "turn": turn})

        # 简单判断对话是否应该结束
        last_content = dialogue[-1]["content"]
        end_keywords = ["挂了", "再见", "拜拜", "稍后再说", "回头聊", "开车不方便", "不方便接"]
        if any(kw in last_content for kw in end_keywords) and dialogue[-1]["role"] == "user":
            break
        if turn >= max_turns:
            break

    return dialogue


if __name__ == "__main__":
    # 测试：生成不同persona的对话
    INSTRUCTION = """# Role
你是美团外卖骑手的站长。

# Task
致电"飞毛腿"骑手，通知他们今天合同已成功签署，并提醒他们完成配送任务。

# Opening Line
你好，请问是${rider_name}吗？我是站长。我看到你已报名飞毛腿。请记住，午餐和晚餐高峰期需要上线。单日合同每天至少完成 **5 单**；多日合同每天至少完成 **3 单**。

# Call Flow
1. 告知骑手今天飞毛腿合同已生效，并询问他们是否可以开始配送。
2. 说明单日飞毛腿合同需要**连续 7 天**完成配送；否则合同将受到影响。
3. 尽量挽留不想配送的骑手，鼓励能配送的骑手，并提醒他们注意安全。
4. 说明飞毛腿报名是按排名进行的，并非站长干预。骑手应减少拒单、取消和超时。

# Constraints
- 每次回复控制在**约 30 个字以内**。
- 保持语气随意，像打电话一样自然。
- 避免重复回复。
- 如果骑手坚持确实无法配送，安慰他们后挂断电话。"""

    for persona_type in ["cooperative", "rejecting", "driving"]:
        print(f"\n{'='*50}")
        print(f"生成对话: persona={persona_type}")
        print(f"{'='*50}")
        dialogue = simulate_dialogue(INSTRUCTION, persona_type)
        for d in dialogue:
            role = "Agent" if d["role"] == "agent" else "User"
            print(f"  [{d['turn']}] {role}: {d['content']}")
