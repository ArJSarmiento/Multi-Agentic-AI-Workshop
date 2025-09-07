from dotenv import load_dotenv
from datetime import datetime, UTC
from pydantic_ai import Agent, RunContext

load_dotenv()

agent = Agent(
    "gemini-2.5-flash",
    instructions="""
You can call `now()` for the current ISO timestamp.
Call it before answering time-sensitive questions.
""",
)


@agent.tool
def now(ctx: RunContext) -> str:
    """
    Returns the current ISO timestamp.
    """
    return datetime.now(UTC).isoformat()


@agent.tool
def echo_with_ctx(ctx: RunContext, msg: str) -> str:
    """
    Returns the echo of the message.
    """
    message_count = len(ctx.messages) if hasattr(ctx, "messages") else 0
    return f"echo={msg} (message_count={message_count})"


def main() -> None:
    print(agent.run_sync("What's the time? Use tools.").output)
    print(agent.run_sync("Call echo_with_ctx with 'hi'.").output)


if __name__ == "__main__":
    main()
