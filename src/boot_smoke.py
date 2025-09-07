from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    agent = Agent("gemini-2.5-flash", instructions="Be concise.")
    res = agent.run_sync("Say 'hello workshop' exactly.")
    print(res.output)


if __name__ == "__main__":
    main()
