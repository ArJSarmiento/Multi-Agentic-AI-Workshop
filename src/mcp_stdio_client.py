import asyncio
from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


calc_server = MCPServerStdio(
    "uv", args=["run", "src/servers/calc_server.py"], timeout=10
)

agent = Agent(
    "gemini-2.5-flash",
    toolsets=[calc_server],
    instructions="Use tools when math calculations or date operations help.",
)


async def main() -> None:
    async with agent:
        res = await agent.run("How many days between 2000-01-01 and 2025-03-18?")
        print(res.output)

        res2 = await agent.run("What is 17 times 23?")
        print(res2.output)


if __name__ == "__main__":
    asyncio.run(main())
