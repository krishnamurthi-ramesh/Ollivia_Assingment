"""
Shared Tool Definitions
Calculator, DateTime, and basic web search tools available to both assistants.
Author: Krishna Murthi
"""

import math
import datetime
import re
import urllib.request
import urllib.parse
import json
from typing import Any, Dict


# ──────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ──────────────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """
    Safe arithmetic calculator.
    Only allows numbers and basic math operators.
    """
    # Whitelist: digits, spaces, operators, parens, decimal points
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\^]+$', expression):
        return "Error: Invalid expression. Only basic arithmetic allowed."
    try:
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        result = eval(expression, {"__builtins__": {}}, {
            "abs": abs, "round": round,
            "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        })
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {str(e)}"


def get_datetime(timezone: str = "UTC") -> str:
    """Returns current date and time."""
    now = datetime.datetime.now()
    utc_now = datetime.datetime.utcnow()
    return (
        f"Current local time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Current UTC time: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def web_search(query: str, max_results: int = 3) -> str:
    """
    Simple DuckDuckGo Instant Answer search (no API key required).
    Falls back gracefully if network is unavailable.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"• {topic['Text'][:200]}")

        if results:
            return "\n".join(results)
        return f"No instant answer found for: {query}"
    except Exception as e:
        return f"Search unavailable: {str(e)}"


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Basic unit converter for common conversions."""
    conversions = {
        # Temperature
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("celsius", "kelvin"): lambda x: x + 273.15,
        ("kelvin", "celsius"): lambda x: x - 273.15,
        # Length
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x / 3.28084,
        ("kilometers", "miles"): lambda x: x * 0.621371,
        ("miles", "kilometers"): lambda x: x / 0.621371,
        # Weight
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x / 2.20462,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."


# ──────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────────────────────────────────────

TOOLS: Dict[str, Dict[str, Any]] = {
    "calculator": {
        "fn": calculator,
        "description": "Evaluate arithmetic expressions. Example: calculator('2 + 2 * 10')",
        "usage": "calculator(<expression>)",
    },
    "datetime": {
        "fn": get_datetime,
        "description": "Get the current date and time.",
        "usage": "datetime()",
    },
    "search": {
        "fn": web_search,
        "description": "Search the web for information using DuckDuckGo.",
        "usage": "search(<query>)",
    },
    "convert": {
        "fn": unit_converter,
        "description": "Convert between units (temperature, length, weight).",
        "usage": "convert(<value>, <from_unit>, <to_unit>)",
    },
}


def run_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"
    return TOOLS[tool_name]["fn"](**kwargs)


def get_tool_descriptions() -> str:
    """Return formatted tool descriptions for system prompt injection."""
    lines = ["Available tools:"]
    for name, info in TOOLS.items():
        lines.append(f"  - {name}: {info['description']}")
    return "\n".join(lines)
