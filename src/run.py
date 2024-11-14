import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import HumanMessage

from src.azure_chat import model
from src.book_car_agent import BookCarAgent
from src.models import BookCar

if __name__ == "__main__":

    # Add memory
    db_path = '../checkpoints/checkpoints.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    slots = BookCar()

    while True:
        user_input = input("User input:\n")

        # Bot
        abot = BookCarAgent(model, checkpointer=memory, slots=slots)
        messages = [HumanMessage(content=user_input)]
        thread = {"configurable": {"thread_id": "1"}}
        result = abot.graph.invoke({"messages": messages}, thread)

        # Print out the AI message
        print("SLOTS: ", result['slots'])
        print("AI Message:", result['messages'][-1].content)

        # Update slots
        slots = result['slots']