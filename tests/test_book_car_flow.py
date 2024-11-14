import sqlite3
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.azure_chat import model
from src.book_car_agent import BookCarAgent
from src.models import BookCar

db_path = '../checkpoints/checkpoints.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)


class TestBookCarFlows:
    def test_in_line_slot_collection(self):
        """Test simple in-line slot collection"""
        slots = BookCar()
        abot = BookCarAgent(model, checkpointer=memory, slots=slots)
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        user_input = ("Book a car in Seattle for 16/01/2026 6pm Drop off San Francisco on 20/01/2026 9am. "
                      "No preferred car")
        expected_slots = {
            'pick_up_location': 'Seattle',
            'pick_up_date': '16/Jan/2026',
            'pick_up_time': '6pm',
            'drop_off_location': 'San Francisco',
            'drop_off_date': '20/Jan/2026',
            'drop_off_time': '9am',
            'car_type': None,
            'preferred_vendor': None,
            'car_loyalty_number': None,
        }

        messages = [HumanMessage(content=user_input)]
        result = abot.graph.invoke({"messages": messages}, thread)

        assert result['slots'].dict() == expected_slots

    def test_sequential_slot_collection(self):
        """Test sequential slot collection"""
        slots = BookCar()
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        user_inputs = ["I need to book a car",
                       "Pick up London 16/01/2026 at 9am",
                       "Drop off Oxford 18/01/2026 at 9am. No car preference."]

        expected_slots = {
            'pick_up_location': 'London',
            'pick_up_date': '16/Jan/2026',
            'pick_up_time': '9am',
            'drop_off_location': 'Oxford',
            'drop_off_date': '18/Jan/2026',
            'drop_off_time': '9am',
            'car_type': None,
            'preferred_vendor': None,
            'car_loyalty_number': None,
        }

        for i, message in enumerate(user_inputs):
            abot = BookCarAgent(model, checkpointer=memory, slots=slots)

            messages = [HumanMessage(content=message)]
            result = abot.graph.invoke({"messages": messages}, thread)
            slots = result['slots']

        assert result['slots'].dict() == expected_slots

