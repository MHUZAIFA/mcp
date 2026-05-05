from flask import Flask, jsonify, request
from mcp import ClientSession
from mcp.client.sse import sse_client
from groq import Groq
import traceback
import json
import os

app = Flask(__name__)
MCP_SERVER_URL = "http://localhost:3000/sse"
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def sanitize_name(name: str) -> str:
    """Replace hyphens with underscores — LLaMA doesn't like hyphens in tool names"""
    return name.replace("-", "_")


async def ask_mcp(prompt: str) -> str:
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()

            # Map sanitized names back to original MCP tool names
            name_map = {sanitize_name(tool.name): tool.name for tool in tools_result.tools}

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": sanitize_name(tool.name),  # ✅ underscores instead of hyphens
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in tools_result.tools
            ]

            messages = [{"role": "user", "content": prompt}]

            response = groq_client.chat.completions.create(
                model="llama3-groq-70b-8192-tool-use-preview",  # ✅ tool-use optimized model
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            while response.choices[0].finish_reason == "tool_calls":
                tool_call = response.choices[0].message.tool_calls[0]
                sanitized_name = tool_call.function.name
                original_name = name_map.get(sanitized_name, sanitized_name)  # ✅ map back
                tool_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Tool called: {original_name}")
                print(f"📥 Args: {tool_args}")

                tool_result = await session.call_tool(original_name, tool_args)
                tool_text = tool_result.content[0].text

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
                    model="llama3-groq-70b-8192-tool-use-preview",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
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