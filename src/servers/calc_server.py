from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Calculator")


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
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - start).days


if __name__ == "__main__":
    mcp.run(transport="stdio")
