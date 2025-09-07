from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

load_dotenv()

# Specialist agents
math_agent = Agent(
    "gemini-2.5-flash",
    instructions="Compute or reason step-by-step; output the final number.",
)
qa_agent = Agent("gemini-2.5-flash", instructions="Answer factual questions concisely.")


# Router output choices (functions are selectable outputs)
async def hand_off_to_math(ctx: RunContext, query: str) -> str:
    """
    If the query is numeric/math/code-like, use hand_off_to_math.
    """
    print("Using math agent")
    messages = ctx.messages[:-1]
    res = await math_agent.run(query, message_history=messages)
    return res.output


async def hand_off_to_qa(ctx: RunContext, query: str) -> str:
    """
    If general knowledge, use hand_off_to_qa.
    """
    print("Using qa agent")
    messages = ctx.messages[:-1]
    res = await qa_agent.run(query, message_history=messages)
    return res.output


class RouterFailure(BaseModel):
    """
    If the query is not numeric/math/code-like or general knowledge, return RouterFailure with an explanation.
    """

    explanation: str


RouterOut = str | RouterFailure

router = Agent[None, RouterOut](
    "gemini-2.5-flash",
    output_type=[hand_off_to_math, hand_off_to_qa, RouterFailure],
    instructions=(
        "If the query is numeric/math/code-like, use hand_off_to_math. "
        "If general knowledge, use hand_off_to_qa. "
        "Otherwise, return RouterFailure with an explanation."
    ),
)


def demo() -> None:
    print("MATH:", router.run_sync("What is 17 * 23?").output)
    print("QA:", router.run_sync("Who wrote The Hobbit?").output)
    print("FALLBACK:", router.run_sync("Tell me something only you know.").output)


if __name__ == "__main__":
    demo()
