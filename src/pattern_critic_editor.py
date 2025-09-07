from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dotenv import load_dotenv

load_dotenv()


class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    suggestions: list[str]


editor = Agent("gemini-2.5-flash", instructions="Draft clearly. Avoid fluff.")
critic = Agent[None, Review](
    "gemini-2.5-flash",
    output_type=Review,
    instructions="Score 1-10; include concrete revision suggestions.",
)


def improve(prompt: str, rounds: int = 2, target: int = 8) -> str:
    """
    Improve the draft by the critic's suggestions.
    """
    draft = editor.run_sync(prompt).output
    for _ in range(rounds):
        review = critic.run_sync(
            f"Rate this draft and suggest edits:\n\n{draft}"
        ).output
        if review.score >= target:
            break
        revision_instructions = "Apply these improvements:\n- " + "\n- ".join(
            review.suggestions
        )
        draft = editor.run_sync(
            f"{prompt}\n\n{revision_instructions}\n\nRevise:"
        ).output

    return draft


if __name__ == "__main__":
    print(improve("Write a 120-word product blurb for a privacy-first notes app."))
