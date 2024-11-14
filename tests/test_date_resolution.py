import sqlite3
import uuid
from datetime import datetime, timedelta

import pytest
from dateutil.relativedelta import relativedelta
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.azure_chat import model
from src.book_car_agent import BookCarAgent
from src.date_validation import DATE_FORMAT
from src.models import BookCar
from tests.helpers import resolve_date_from_day_of_week

db_path = '../checkpoints/checkpoints.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)


class TestDateResolution:

    @pytest.mark.parametrize(
        "date_value, expected",
        [
            ("sdasdasdsasda", "INVALID. Reason: Date needs clarification."),
            ("yesterday", "INVALID. Reason: Date is in the past."),
            ("31/Jun/2028", "INVALID. Reason: Date needs clarification."),  # June has only 30 days
            ("June 31st", "INVALID. Reason: Date needs clarification."),  # June has only 30 days
            # Standard date formats
            ("30/Jun/2028", "30/Jun/2028"),
            ("30/6/2028", "30/Jun/2028"),
            ("30/06/2028", "30/Jun/2028"),
            ("30-06-2028", "30/Jun/2028"),
            ("30.06.2028", "30/Jun/2028"),
            ("2028-06-30", "30/Jun/2028"),
            ("June 30 2028", "30/Jun/2028"),
            ("30th June 2028", "30/Jun/2028"),
        ],
    )
    def test_standard_date_formats(self, date_value, expected):
        slots = BookCar()
        abot = BookCarAgent(model, checkpointer=memory, slots=slots)
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        user_input = f"Book a car in Seattle, pick up date {date_value}"
        expected_slots = {
            'pick_up_location': 'Seattle',
            'pick_up_date': expected,
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

    @pytest.mark.parametrize(
        "date_value, expected",
        [
            # Basic relative dates
            ("today", datetime.now().strftime(DATE_FORMAT)),
            ("tomorrow", (datetime.now() + timedelta(days=1)).strftime(DATE_FORMAT)),
            ("day after tomorrow", (datetime.now() + timedelta(days=2)).strftime(DATE_FORMAT)),
            # Days-based relative dates
            ("in 2 days", (datetime.now() + timedelta(days=2)).strftime(DATE_FORMAT)),
            ("in 3 days", (datetime.now() + timedelta(days=3)).strftime(DATE_FORMAT)),
            ("in 5 days", (datetime.now() + timedelta(days=5)).strftime(DATE_FORMAT)),
            ("in a week", (datetime.now() + timedelta(days=7)).strftime(DATE_FORMAT)),
            # Week-based relative dates
            ("next week", (datetime.now() + timedelta(days=7)).strftime(DATE_FORMAT)),
            ("in 2 weeks", (datetime.now() + timedelta(days=14)).strftime(DATE_FORMAT)),
            ("in 3 weeks", (datetime.now() + timedelta(days=21)).strftime(DATE_FORMAT)),
            # Month-based relative dates
            ("next month", (datetime.now() + relativedelta(months=1)).strftime(DATE_FORMAT)),
            ("in 2 months", (datetime.now() + relativedelta(months=2)).strftime(DATE_FORMAT)),
            ("in 3 months", (datetime.now() + relativedelta(months=3)).strftime(DATE_FORMAT)),
            # Weekday references
            ("Monday", resolve_date_from_day_of_week("Monday")),
            ("Tuesday", resolve_date_from_day_of_week("Tuesday")),
            ("Wednesday", resolve_date_from_day_of_week("Wednesday")),
            ("Thursday", resolve_date_from_day_of_week("Thursday")),
            ("Friday", resolve_date_from_day_of_week("Friday")),
            ("Saturday", resolve_date_from_day_of_week("Saturday")),
            ("Sunday", resolve_date_from_day_of_week("Sunday")),
            # Explicit dates that should roll to next year
            ("23/1", datetime(year=datetime.now().year + 1, month=1, day=23).strftime(DATE_FORMAT)),
            ("January 15", datetime(year=datetime.now().year + 1, month=1, day=15).strftime(DATE_FORMAT)),
            ("15th January", datetime(year=datetime.now().year + 1, month=1, day=15).strftime(DATE_FORMAT)),
            ("February 1st", datetime(year=datetime.now().year + 1, month=2, day=1).strftime(DATE_FORMAT)),
            ("1st February", datetime(year=datetime.now().year + 1, month=2, day=1).strftime(DATE_FORMAT)),
            ("March 1", datetime(year=datetime.now().year + 1, month=3, day=1).strftime(DATE_FORMAT)),
            ("1st of March", datetime(year=datetime.now().year + 1, month=3, day=1).strftime(DATE_FORMAT)),
            ("April 15", datetime(year=datetime.now().year + 1, month=4, day=15).strftime(DATE_FORMAT)),
        ],
    )
    def test_relative_date_resolution(self, date_value, expected):
        slots = BookCar()
        abot = BookCarAgent(model, checkpointer=memory, slots=slots)
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        user_input = f"Book a car in Seattle, pick up date {date_value}"
        expected_slots = {
            'pick_up_location': 'Seattle',
            'pick_up_date': expected,
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
