from flask import Flask, jsonify, request
from mcp import ClientSession
from mcp.client.sse import sse_client
from groq import Groq
import traceback

app = Flask(__name__)
MCP_SERVER_URL = "http://localhost:3000/sse"
groq_client = Groq(api_key="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")


async def ask_mcp(prompt: str) -> str:
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in tools_result.tools
            ]

            messages = [{"role": "user", "content": prompt}]

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # free & supports tool use
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            # Handle tool calls in a loop
            while response.choices[0].finish_reason == "tool_calls":
                tool_call = response.choices[0].message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = __import__("json").loads(tool_call.function.arguments)

                # Call the tool on the MCP server
                tool_result = await session.call_tool(tool_name, tool_args)
                tool_text = tool_result.content[0].text

                print(f"###Tool called: {tool_name}")  # 👈 add this
                print(f"####Args: {tool_args}")
                print(f"####Result preview: {tool_text[:100]}...")

                # Append assistant + tool result to messages
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