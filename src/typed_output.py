from dotenv import load_dotenv
from pydantic_ai import Agent
from models.answer_schema import Typed, Answer, Fallback

load_dotenv()

agent = Agent[None, Typed](
    "gemini-2.5-flash",
    output_type=Answer | Fallback,  # type: ignore[valid-type]
    instructions="Return a factual Answer model; if unsure, return Fallback.",
)


def main() -> None:
    result1 = agent.run_sync("What is the capital of France?").output
    print(result1.model_dump_json(indent=2))

    result2 = agent.run_sync("Gibberish 123??").output
    print(result2.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
