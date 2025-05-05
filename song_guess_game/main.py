#!/usr/bin/env python3
import argparse
import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableSerializable
from langchain_core.chat_history import InMemoryChatMessageHistory

# ─── globals ───────────────────────────────────────────────────
history_store = InMemoryChatMessageHistory()


def build_stateless_chain(temperature: float) -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a music expert. Given a short fragment of song" +
                "lyrics or a description, guess the song title and artist" +
                "in one concise line.",
            ),
            ("human", "{fragment}"),
        ]
    )
    return prompt | ChatOpenAI(temperature=temperature)


def build_stateful_chain(temperature: float) -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a music expert playing a guessing game with" +
                "the user. Use what we already discussed to improve" +
                "future guesses."
            ),
            MessagesPlaceholder("history"),
            ("human", "{fragment}"),
        ]
    )

    def get_history(_="game"):
        return history_store

    return RunnableWithMessageHistory(
        prompt | ChatOpenAI(temperature=temperature),
        get_session_history=get_history,
        input_messages_key="fragment",
        history_messages_key="history",
    )


# ─── repl ──────────────────────────────────────────────────────
def repl(memory_on: bool, temperature: float):
    print("🎵  Guess the Song  –  type 'exit' to quit.")
    print("Commands:  temp <value>\n")
    chain = \
        build_stateful_chain(temperature=temperature) \
        if memory_on else build_stateless_chain(temperature=temperature)

    while True:
        try:
            user_in = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_in:
            continue
        cmd = user_in.lower()

        if cmd in {"exit", "quit"}:
            print("Bye.")
            break

        if cmd.startswith("temp "):
            try:
                global TEMPERATURE
                TEMPERATURE = float(cmd.split()[1])
                print(f"[Temperature set to {TEMPERATURE}]")
                # rebuild chain with new temperature
                chain = build_stateful_chain(temperature=temperature) \
                    if memory_on else \
                    build_stateless_chain(temperature=temperature)
            except (IndexError, ValueError):
                print("Usage: temp 0.3")
            continue

        if memory_on:
            ai_msg = chain.invoke(
                {"fragment": user_in},
                config={"configurable": {"session_id": "game"}},
            )
        else:
            ai_msg = chain.invoke({"fragment": user_in})

        print("Bot >", ai_msg.content)


# ─── entry ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp",
                        type=float, help="Initial model temperature")
    args = parser.parse_args()

    if not args.temp:
        args.temp = 0.05

    if not os.getenv("OPENAI_API_KEY"):
        print("❌  Please set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # one‑time choice for memory before starting the loop
    memory_flag = True
    while True:
        choice = input("Enable memory? (on/off) > ").strip().lower()
        if choice in {"on", "off"}:
            memory_flag = choice == "on"
        print("default to on.")
        break

    repl(memory_flag, temperature=args.temp)


if __name__ == "__main__":
    main()
