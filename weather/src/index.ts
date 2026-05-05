import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const NWS_API_BASE = "https://api.weather.gov";
const USER_AGENT = "weather-app/1.0";

const server = new McpServer(
  {
    name: "Weather",
    description: "Provides weather information for a given location.",
    version: "1.0.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {}
    },
  },
);
