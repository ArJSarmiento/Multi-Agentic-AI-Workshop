from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


class Requirements(BaseModel):
    topic: str
    audience: str
    length_words: int = Field(ge=50, le=800)


extractor = Agent[None, Requirements](
    "gemini-2.5-flash",
    output_type=Requirements,
    instructions="Extract topic, audience, and a reasonable length (50-800).",
)


class Outline(BaseModel):
    headings: list[str]


outliner = Agent[None, Outline](
    "gemini-2.5-flash",
    output_type=Outline,
    instructions="Produce 3-6 descriptive headings.",
)

drafter = Agent(
    "gemini-2.5-flash", instructions="Write a crisp draft under the provided headings."
)


def pipeline_run(user_brief: str) -> str:
    req = extractor.run_sync(user_brief).output
    outline = outliner.run_sync(
        f"Topic: {req.topic}\nAudience: {req.audience}\nLength: {req.length_words}"
    ).output
    draft = drafter.run_sync(
        "Write with these headings:\n"
        + "\n".join(f"- {h}" for h in outline.headings)
        + f"\nTarget length ~{req.length_words} words."
    ).output
    return draft


if __name__ == "__main__":
    print(
        pipeline_run(
            "I need a short blog about zero-copy networking for backend engineers."
        )
    )
