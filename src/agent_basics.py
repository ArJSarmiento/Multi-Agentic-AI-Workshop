import asyncio
import nest_asyncio
from pydantic_ai import Agent
from dotenv import load_dotenv
import sys

nest_asyncio.apply()

load_dotenv()

agent = Agent("gemini-2.5-flash", instructions="Answer briefly.")


def run_sync_demo() -> None:
    res = agent.run_sync("Name three prime numbers.")
    print("SYNC:", res.output)


async def run_async_demo() -> None:
    res = await agent.run("One sentence on Fibonacci numbers.")
    print("ASYNC:", res.output)


async def run_stream_demo() -> None:
    async with agent.run_stream(
        "Stream a short paragraph on solar eclipses."
    ) as stream:
        async for text_chunk in stream.stream_text():
            print(text_chunk, end="", flush=True)

        print("\n---")
        final = await stream.get_output()
        print("FINAL:", final)


async def run_all_async_demos() -> None:
    """Run all async demos in a single event loop"""
    await run_async_demo()
    await run_stream_demo()


if __name__ == "__main__":
    # Set event loop policy for better compatibility
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    run_sync_demo()

    # Run all async demos in a single event loop to avoid conflicts
    asyncio.run(run_all_async_demos())
