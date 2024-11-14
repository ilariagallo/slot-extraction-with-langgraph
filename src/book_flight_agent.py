from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import Agent
from src.date_validation import date_parser, validate_timeline
from src.models import BookFlight


class BookFlightAgent(Agent):
    """Agent for new flight booking"""

    optional_slots_keys = ['preferred_airline', 'miles_loyalty_number']

    def __init__(self, model: AzureChatOpenAI, slots: BookFlight, checkpointer: SqliteSaver, optional_slots_keys=None):
        super().__init__(
            model, slots, checkpointer, optional_slots_keys=optional_slots_keys or self.optional_slots_keys
        )

    def validate_slots(self, slots) -> BookFlight:
        parsed_output = date_parser(slots)
        slots = validate_timeline(parsed_output, 'departure_date', 'arrival_date')
        return slots
