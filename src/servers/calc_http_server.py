from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Calculator-HTTP")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def days_between(start_date: str, end_date: str) -> int:
    """Calculate days between two dates (YYYY-MM-DD format)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - start).days


@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number."""
    if n < 0:
        return 0
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # http://localhost:8000/mcp
