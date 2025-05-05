from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# model used for prediction.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# model used for summarization.
summarize = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

memory = ConversationSummaryBufferMemory(
    llm=summarize, max_token_limit=100, memory_key="history", return_messages=True
)

# Chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an animal‑guessing bot. Ask yes/no questions to identify the "
     "animal the user has in mind. When you're sure, propose an answer."),
    MessagesPlaceholder("history"),   # raw 2 turns + summary up top
    ("assistant", "Ask your next question."),
])

def get_history(_sid="game"):
    return memory.chat_memory

if __name__ == "__main__":
    chain = RunnableWithMessageHistory(
        runnable=prompt | llm,
        get_session_history=get_history,
        input_messages_key="user_input",
        history_messages_key="history",
    )

    print("Think of an animal. I'll try to guess it!")
    while True:
        user = input("You > ").strip()
        if user.lower() in {"quit", "exit"}:
            break

        response = chain.invoke(
            {"user_input": user},
            config={"configurable": {"session_id": "game"}},
        )
        memory.save_context({"input": user.lower()},
                            {"output": response.content})

        print("Summary> ", memory.moving_summary_buffer)
        print("Bot >", response.content)

        if "yes" in response.content:
            print("Yay! 🎉")
            break
