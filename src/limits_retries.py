import random
from time import sleep
from tenacity import retry, wait_exponential, stop_after_attempt
from pydantic_ai import Agent, UsageLimits, RunContext
from dotenv import load_dotenv

load_dotenv()

agent = Agent("gemini-2.5-flash", instructions="Be brief; avoid verbosity.")


# Simulate a flaky HTTP call via a tool; Tenacity handles retries/backoff
@agent.tool
@retry(
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2), stop=stop_after_attempt(5)
)
def flaky_fetch(ctx: RunContext, q: str) -> str:
    if random.random() < 0.6:
        # Simulate transient failure
        sleep(0.05)
        raise RuntimeError("Transient network error")

    return f"data-for:{q}"


def main() -> None:
    limits = UsageLimits(request_limit=4, tool_calls_limit=3, total_tokens_limit=1500)
    res = agent.run_sync(
        "Call flaky_fetch with 'hello' then summarize result.", usage_limits=limits
    )
    print(res.output)


if __name__ == "__main__":
    main()
