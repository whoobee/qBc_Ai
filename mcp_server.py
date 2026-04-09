import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime

logger = logging.getLogger("qBc_Ai.mcp")

class McpServer:
    """Local Tool Server simulating the Model Context Protocol (MCP)."""

    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current local time and date.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a specified city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g., 'London' or 'New York'"
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    }
                }
            }
        ]

    def get_tools_schema(self):
        return self.tools

    def execute_tool(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
            logger.info("Executing tool '%s' with args %s", name, args)

            if name == "get_time":
                now = datetime.now()
                return now.strftime("%A, %B %d, %Y %I:%M %p")

            elif name == "get_weather":
                location = args.get("location", "Unknown")
                location_encoded = urllib.parse.quote(location)
                url = f"https://wttr.in/{location_encoded}?format=3"
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.68.0'})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        if response.status == 200:
                            return response.read().decode('utf-8').strip()
                        return f"Weather for {location} is currently unavailable."
                except Exception as e:
                    return f"Error fetching weather: {e}"

            else:
                return f"Unknown tool: {name}"
                
        except Exception as e:
            logger.error("Tool execution failed: %s", e)
            return f"Error executing {name}: {e}"
