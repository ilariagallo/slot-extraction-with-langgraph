import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import HumanMessage

from src.graph_without_tool_calling.azure_chat import model
from src.graph_without_tool_calling.agent import Agent
from src.graph_without_tool_calling.models import BookCar, BOOK_CAR_OPTIONAL_SLOTS

if __name__ == "__main__":

    # Add memory
    db_path = '../checkpoints/checkpoints.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    slots = BookCar()
    optional_slots_keys = BOOK_CAR_OPTIONAL_SLOTS

    while True:
        user_input = input("User input:\n")

        # Bot
        abot = Agent(model, checkpointer=memory, slots=slots, optional_slots_keys=optional_slots_keys)
        messages = [HumanMessage(content=user_input)]
        thread = {"configurable": {"thread_id": "1"}}
        result = abot.graph.invoke({"messages": messages}, thread)

        # Print out the AI message
        print("SLOTS: ", result['slots'])
        print("AI Message:", result['messages'][-1].content)

        # Update slots
        slots = result['slots']