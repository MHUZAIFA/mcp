from flask import Flask, jsonify, request
from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio

app = Flask(__name__)
MCP_SERVER_URL = "http://localhost:3000/sse"


async def call_tool(tool_name: str, arguments: dict):
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content[0].text


def parse_forecast(raw: str) -> dict:
    lines = raw.strip().split("\n")
    # First line is "Forecast for lat, lon:"
    header = lines[0].replace("Forecast for ", "").replace(":", "").strip()
    lat, lon = header.split(", ")

    periods = []
    current = {}
    for line in lines[2:]:  # skip header and blank line
        if line == "---":
            if current:
                periods.append(current)
                current = {}
        elif line.startswith("Temperature:"):
            current["temperature"] = line.replace("Temperature: ", "").strip()
        elif line.startswith("Wind:"):
            current["wind"] = line.replace("Wind: ", "").strip()
        elif ":" in line and not any(k in current for k in ["name"]):
            current["name"] = line.replace(":", "").strip()
        else:
            current["condition"] = line.strip()

    return {"location": {"latitude": lat, "longitude": lon}, "periods": periods}


def parse_alerts(raw: str) -> dict:
    if raw.startswith("No active alerts"):
        return {"alerts": [], "message": raw}

    blocks = raw.strip().split("---\n")
    alerts = []
    for block in blocks:
        if not block.strip():
            continue
        alert = {}
        for line in block.strip().split("\n"):
            if ": " in line:
                key, val = line.split(": ", 1)
                alert[key.lower()] = val.strip()
        if alert:
            alerts.append(alert)

    return {"alerts": alerts}


@app.route("/forecast")
async def forecast():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400
    try:
        raw = await call_tool("get-forecast", {"latitude": lat, "longitude": lon})
        return jsonify(parse_forecast(raw))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/alerts")
async def alerts():
    state = request.args.get("state", "").upper()
    if not state:
        return jsonify({"error": "state is required"}), 400
    try:
        raw = await call_tool("get-alerts", {"state": state})
        return jsonify(parse_alerts(raw))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")