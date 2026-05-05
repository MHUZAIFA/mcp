from flask import Flask, jsonify, request
from mcp import ClientSession
from mcp.client.sse import sse_client
from groq import Groq
import traceback
import json
import os


def load_local_env() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env.local")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


app = Flask(__name__)
MCP_SERVER_URL = "http://localhost:3000/sse"
MAX_TOOL_CALLS = 5
MAX_TOOL_RESPONSE_CHARS = 1500

load_local_env()

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to client/.env.local or your environment.")

groq_client = Groq(api_key=groq_api_key)


def sanitize_name(name: str) -> str:
    """Replace hyphens with underscores — LLaMA doesn't like hyphens in tool names"""
    return name.replace("-", "_")


async def ask_mcp(prompt: str) -> str:
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            name_map = {sanitize_name(tool.name): tool.name for tool in tools_result.tools}

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": sanitize_name(tool.name),
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in tools_result.tools
            ]

            messages = [{"role": "user", "content": prompt}]
            tool_call_count = 0
            called_tools = set()

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            while response.choices[0].finish_reason == "tool_calls" and tool_call_count < MAX_TOOL_CALLS:
                tool_call_count += 1
                tool_call = response.choices[0].message.tool_calls[0]
                sanitized_name = tool_call.function.name
                original_name = name_map.get(sanitized_name, sanitized_name)
                tool_args = json.loads(tool_call.function.arguments)

                # ✅ skip duplicate tool calls — but force a final answer first
                tool_key = f"{original_name}:{json.dumps(tool_args, sort_keys=True)}"
                if tool_key in called_tools:
                    print(f"⏭️ Skipping duplicate, forcing final answer...")
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        tools=tools,
                        tool_choice="none",
                    )
                    break
                called_tools.add(tool_key)

                print(f"🔧 [{tool_call_count}/{MAX_TOOL_CALLS}] Tool called: {original_name}")
                print(f"📥 Args: {tool_args}")

                tool_result = await session.call_tool(original_name, tool_args)
                tool_text = tool_result.content[0].text

                if len(tool_text) > MAX_TOOL_RESPONSE_CHARS:
                    tool_text = tool_text[:MAX_TOOL_RESPONSE_CHARS] + "\n...[truncated]"

                print(f"📤 Result preview: {tool_text[:100]}...")

                messages += [
                    response.choices[0].message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_text,
                    }
                ]

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )

            if tool_call_count >= MAX_TOOL_CALLS:
                print("⚠️ Max tool calls reached, forcing final answer")
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tool_choice="none",
                )

            return response.choices[0].message.content


@app.route("/ask")
async def ask():
    prompt = request.args.get("q")
    if not prompt:
        return jsonify({"error": "q parameter is required"}), 400
    try:
        answer = await ask_mcp(prompt)
        return jsonify({"response": answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")