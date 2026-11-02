"""Terminal conversation and approvals, backed by the same durable agent loop."""

import json
import threading

from domain import reply_text

HELP = """/new         开始新会话
/sessions    列出会话
/use ID      切换到已有会话
/resume      继续中断的任务
/status      查看会话状态和后台进展
/goal 条件   设置目标，/goal 查看，/goal clear 清除
/exit        保存会话并退出"""


def _open_session(service, sid):
    state = service.view(sid)
    print(f"\n会话 {sid}")
    for msg in reversed(state["messages"]):
        if msg["role"] == "assistant" and (text := reply_text(msg)):
            print(f"上次回复 › {text}")
            break
    return len(state["messages"])


def chat(service, session=None):
    """Run until /exit, EOF or interrupt. Leave outstanding approvals pending on exit."""
    sid = session or service.new_session()
    print(f"agent · {service.workspace}    /help 查看命令")
    seen = _open_session(service, sid)
    foreground = threading.current_thread()

    def show_tool(name, args, _result):
        if threading.current_thread() is foreground:
            path = args.get("path")
            print(f"  · {name}" + (f" {path!r}" if path else ""), flush=True)

    service.hooks.on("PostToolUse", show_tool)
    try:
        while True:
            state = service.view(sid)
            for msg in state["messages"][seen:]:
                if msg["role"] == "assistant" and (text := reply_text(msg)):
                    print(f"\nAgent › {text}", flush=True)
            seen = len(state["messages"])
            try:
                if state["approvals"]:
                    req = state["approvals"][0]
                    args = json.dumps(req["arguments"], ensure_ascii=False)
                    print(f"\n请求审批 · {req['actor']} · {req['name']} {args}")
                    choice = input("允许执行？[y/N] ").strip().lower()
                    if choice == "/exit":
                        return
                    if choice not in {"", "n", "no", "y", "yes"}:
                        print("请输入 y 或 n。")
                        continue
                    result = service.approve(sid, req["id"], choice in {"y", "yes"})
                else:
                    line = input("\n你 › ").strip()
                    if not line:
                        continue
                    if line == "/exit":
                        return
                    if line == "/help":
                        print(HELP)
                        continue
                    if line == "/new":
                        sid = service.new_session()
                        seen = _open_session(service, sid)
                        continue
                    if line == "/sessions":
                        print("\n".join(service.list_sessions()))
                        continue
                    if line.startswith("/use "):
                        target = line[5:].strip()
                        seen = _open_session(service, target)
                        sid = target
                        continue
                    if line == "/status":
                        print(f"状态：{service.view(sid)['status']}")
                        continue
                    print("处理中…", flush=True)
                    result = service.resume(sid) if line == "/resume" else service.chat(sid, line)
                status = result["status"]
                if status == "waiting":
                    print("后台任务运行中，按回车查看进展。")
                elif status not in {"done", "approval", "recorded"}:
                    print(f"[{status}] {result.get('text', '')}")
            except (KeyError, ValueError) as exc:
                print(f"请求失败：{exc}")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        service.hooks.callbacks["PostToolUse"].remove(show_tool)
