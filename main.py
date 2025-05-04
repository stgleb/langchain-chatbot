import os
import sys

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain, LLMChain, SequentialChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAI


def simple_model_call():
    llm = OpenAI(temperature=0)

    response = llm.invoke(input="Explain quantum computing in one sentence.")

    print(f"LLM Response: {response}")


def prompt_template_example():
    llm = OpenAI(temperature=0.7)

    template = "Write a {adjective} poem about {subject}."
    prompt_template = PromptTemplate(
        input_variables=["adjective", "subject"],
        template=template,
    )

    chain = prompt_template | llm
    response = chain.invoke(input={"adjective": "funny", "subject": "programming"})

    print(f"Generated Prompt: {template.format(adjective='funny', subject='programming')}")
    print(f"LLM Response: {response}")


def chain_example():
    # Initialize the OpenAI LLM
    llm = OpenAI(temperature=0.7)

    prompt_template1 = PromptTemplate(
        input_variables=["animal"],
        template="Generate 3 names for a {animal} character in a children's book."
    )

    prompt_template2 = PromptTemplate(
        input_variables=["names"],
        template="Select the best name from these options and create a short story about this character: {names}"
    )

    chain1 = prompt_template1 | llm
    chain2 = prompt_template2 | llm
    sequential_chain = chain1 | chain2

    result = sequential_chain.invoke(input={"animal": "dolphin"})
    print(f"Chain Result: {result}")


def calculator(query: str) -> str:
    query = query.strip()

    if "+" in query:
        operation = "+"
        parts = query.split("+")
        if len(parts) != 2:
            return "Invalid input format for addition"
        try:
            a = float(parts[0].strip())
            b = float(parts[1].strip())
            result = a + b
        except ValueError:
            return "Invalid numbers for calculation"
    elif "-" in query:
        operation = "-"
        parts = query.split("-")
        if len(parts) != 2:
            return "Invalid input format for subtraction"
        try:
            a = float(parts[0].strip())
            b = float(parts[1].strip())
            result = a - b
        except ValueError:
            return "Invalid numbers for calculation"
    else:
        return "Unsupported operation"

    return f"{a} {operation} {b} = {result}"


def agent_with_tools_example():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for performing calculations"
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    with get_openai_callback() as cb:
        response = agent.invoke(input="What is 25.5 + 10.8?")
        print(f"Agent Response: {response}")
        print(f"Total Tokens: {cb.total_tokens}")


def memory_example():
    llm = ChatOpenAI(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Store for chat histories
    store = {}

    # Function to get or create chat history for a session
    def get_by_session_id(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    pipeline = RunnableWithMessageHistory(
        runnable=prompt | llm,
        get_session_history=get_by_session_id,
        input_messages_key="input",
        history_messages_key="history"
    )

    response1 = pipeline.invoke(
        {"input": "Hi, my name is Bob. What's your name?"},
        config={"configurable": {"session_id": "foo"}}
    )
    print(f"AI: {response1.content}")

    response2 = pipeline.invoke(
        {"input": "What did I just tell you my name was?"},
        config={"configurable": {"session_id": "foo"}}
    )
    print(f"AI: {response2.content}")

    response3 = pipeline.invoke(
        {"input": "Tell me about yourself."},
        config={"configurable": {"session_id": "foo"}}
    )
    print(f"AI: {response3.content}")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable must be set")
        sys.exit(1)
    #
    # # Example 1: Simple LLM call
    print("=== Example 1: Simple LLM Call ===")
    simple_model_call()
    #
    # # Example 2: Using a prompt template
    print("\n=== Example 2: Using a Prompt Template ===")
    prompt_template_example()

    # # Example 3: Using a Chain
    print("\n=== Example 3: Using a Chain ===")
    chain_example()

    # # Example 4: Using an Agent with Tools
    print("\n=== Example 4: Using an Agent with Tools ===")
    agent_with_tools_example()

    # Example 5: Using Memory for Conversation
    print("\n=== Example 5: Using Memory for Conversation ===")
    memory_example()