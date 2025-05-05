#!/usr/bin/env python3
import os
import sys

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

from animal_guess.main import prompt

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("export OPENAI_API_KEY first")

# model that chats with the user
chat_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# model that rewrites overflow into a concise summary
summarise_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

memory = ConversationSummaryBufferMemory(
    llm=summarise_llm,   # the summariser
    max_token_limit=100, # after 100 tokens of raw history → summarise
    return_messages=True # keep messages as BaseMessage objects
)

chain = ConversationChain(llm=chat_llm, memory=memory, verbose=False)

if __name__ == "__main__":
    print("🤖  Chat – type 'exit' to quit")
    while True:
        user = input("You > ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        ai_response = chain.predict(input=user)
        print("Bot >", ai_response)

        # show the running summary after every exchange
        print("\n[Summary so far]")
        print(memory.moving_summary_buffer or "(nothing summarised yet)")
        print("-" * 40)
