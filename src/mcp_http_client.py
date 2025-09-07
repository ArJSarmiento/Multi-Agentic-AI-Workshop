import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from dotenv import load_dotenv

load_dotenv()

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")
agent = Agent("gemini-2.5-flash", toolsets=[server])


async def main() -> None:
    async with agent:
        res = await agent.run("What is 7 plus 5? Use the tool.")
        print(res.output)

        res2 = await agent.run("Calculate the factorial of 6.")
        print(res2.output)


if __name__ == "__main__":
    asyncio.run(main())
