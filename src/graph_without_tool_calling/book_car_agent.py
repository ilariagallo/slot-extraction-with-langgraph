from src.graph_without_tool_calling.agent import Agent
from src.graph_without_tool_calling.date_validation import date_parser, validate_pick_up_drop_off_dates
from src.graph_without_tool_calling.models import BookCar


class BookCarAgent(Agent):
    """Agent for new car booking"""
    def validate_slots(self, slots) -> BookCar:
        parsed_output = date_parser(slots)
        slots = validate_pick_up_drop_off_dates(parsed_output)
        return slots

