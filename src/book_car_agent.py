from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import Agent
from src.date_validation import date_parser, validate_timeline
from src.models import BookCar


class BookCarAgent(Agent):
    """Agent for new car booking"""

    optional_slots_keys = ['car_type', 'preferred_vendor', 'car_loyalty_number']

    def __init__(self, model: AzureChatOpenAI, slots: BookCar, checkpointer: SqliteSaver, optional_slots_keys=None):
        super().__init__(
            model, slots, checkpointer, optional_slots_keys=optional_slots_keys or self.optional_slots_keys
        )

    def validate_slots(self, slots) -> BookCar:
        parsed_output = date_parser(slots)
        slots = validate_timeline(parsed_output, 'pick_up_date', 'drop_off_date')
        return slots
