import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime

logger = logging.getLogger("qBc_Ai.mcp")

VALID_EYE_COLORS = [
    "orange", "amber", "purple", "ice", "cyan", "green", "red", "pink", "white"
]

class McpServer:
    """Local Tool Server simulating the Model Context Protocol (MCP)."""

    def __init__(self, mqtt_client=None):
        self._mqtt = mqtt_client
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
            },
            {
                "type": "function",
                "function": {
                    "name": "set_volume",
                    "description": (
                        "Set the robot's global sound volume. "
                        "Accepts a value from 0 (mute) to 100 (maximum)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "volume": {
                                "type": "integer",
                                "description": "Volume level from 0 to 100"
                            }
                        },
                        "required": ["volume"],
                        "additionalProperties": False,
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_eye_color",
                    "description": (
                        "Change the robot's eye color. "
                        f"Available colors: {', '.join(VALID_EYE_COLORS)}."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "color": {
                                "type": "string",
                                "description": (
                                    "Eye color name, one of: "
                                    + ", ".join(VALID_EYE_COLORS)
                                )
                            }
                        },
                        "required": ["color"],
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

            elif name == "set_volume":
                volume = int(args.get("volume", 50))
                volume = max(0, min(100, volume))
                if self._mqtt:
                    self._mqtt.publish(
                        "robot/settings/audio",
                        json.dumps({"global_volume": volume}),
                        qos=1,
                    )
                    return f"Volume set to {volume}%."
                return "MQTT not available — volume not changed."

            elif name == "set_eye_color":
                color = args.get("color", "").strip().lower()
                if color not in VALID_EYE_COLORS:
                    return (
                        f"Unknown color '{color}'. "
                        f"Available colors: {', '.join(VALID_EYE_COLORS)}."
                    )
                if self._mqtt:
                    self._mqtt.publish(
                        "robot/settings/display",
                        json.dumps({"eye_color": color}),
                        qos=1,
                    )
                    return f"Eye color changed to {color}."
                return "MQTT not available — eye color not changed."

            else:
                return f"Unknown tool: {name}"
                
        except Exception as e:
            logger.error("Tool execution failed: %s", e)
            return f"Error executing {name}: {e}"
