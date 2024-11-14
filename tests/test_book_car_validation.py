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


class TestBookCarValidation:
    def test_date_in_the_past(self):
        """Test date in the past is detected as invalid"""
        slots = BookCar()
        abot = BookCarAgent(model, checkpointer=memory, slots=slots)
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        user_input = "Book a car in Seattle for 16/01/2024"
        expected_slots = {
            'pick_up_location': 'Seattle',
            'pick_up_date': 'INVALID. Reason: Date is in the past.',
            'pick_up_time': None,
            'drop_off_location': None,
            'drop_off_date': None,
            'drop_off_time': None,
            'car_type': None,
            'preferred_vendor': None,
            'car_loyalty_number': None,
        }

        messages = [HumanMessage(content=user_input)]
        result = abot.graph.invoke({"messages": messages}, thread)

        assert result['slots'].dict() == expected_slots

    def test_pick_up_after_drop_off_date(self):
        """Test pick up date after drop off date is detected as invalid"""
        slots = BookCar()
        abot = BookCarAgent(model, checkpointer=memory, slots=slots)
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        user_input = "Book a car pick up in Seattle for 16/Jan/2026, drop off in the same location for 10/Jan/2026"
        expected_slots = {
            'pick_up_location': 'Seattle',
            'pick_up_date': "INVALID. Reason: Start date needs to be before the end date.",
            'pick_up_time': None,
            'drop_off_location': 'Seattle',
            'drop_off_date': "INVALID. Reason: Start date needs to be before the end date.",
            'drop_off_time': None,
            'car_type': None,
            'preferred_vendor': None,
            'car_loyalty_number': None,
        }

        messages = [HumanMessage(content=user_input)]
        result = abot.graph.invoke({"messages": messages}, thread)

        assert result['slots'].dict() == expected_slots
