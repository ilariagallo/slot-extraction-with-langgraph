import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import HumanMessage, ToolMessage

from src.tool_calling_approach.azure_chat import model
from src.tool_calling_approach.agent import Agent, PROMPT
from src.tool_calling_approach.tools import book_car

if __name__ == "__main__":

    # Add memory
    db_path = '../checkpoints/checkpoints.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    while True:
        user_input = input("User input:\n")

        # Bot
        abot = Agent(model, [book_car], system=PROMPT, checkpointer=memory)
        messages = [HumanMessage(content=user_input)]
        thread = {"configurable": {"thread_id": "1"}}
        result = abot.graph.invoke({"messages": messages}, thread)

        # Print out the results
        tool_messages = [message for message in result['messages'] if isinstance(message, ToolMessage)]
        if tool_messages:
            current_tool_message = tool_messages[-1].content
            print(current_tool_message)

        agent_message = result['messages'][-1].content
        print(agent_message)
