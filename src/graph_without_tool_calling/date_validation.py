import datetime
from typing import Type

import dateparser

from src.graph_without_tool_calling.models import Schema

DATE_FORMAT = '%d/%b/%Y'


def date_parser(llm_output: Type[Schema]) -> Schema:
    for key, value in llm_output.dict().items():
        if 'date' in key and value:
            parsed_date = dateparser.parse(value, settings={'PREFER_DATES_FROM': 'future'})
            date_str = parsed_date.strftime(DATE_FORMAT) if parsed_date else value
            setattr(llm_output, key, date_str)

    return llm_output


def validate_timeline(parsed_output: Type[Schema], start_date_key: str, end_date_key: str) -> Schema:
    try:
        start_date = getattr(parsed_output, start_date_key)
        end_date = getattr(parsed_output, end_date_key)

        # Try to parse the date in order to validate them
        if start_date and end_date:
            start_date = datetime.datetime.strptime(start_date, DATE_FORMAT)
            end_date = datetime.datetime.strptime(end_date, DATE_FORMAT)

            if start_date > end_date:
                setattr(parsed_output, start_date_key, f"INVALID. Reason: Start date needs to be before the end date.")
                setattr(parsed_output, end_date_key, f"INVALID. Reason: Start date needs to be before the end date.")

        return parsed_output

    except Exception:
        # If parsing fails, validation will not be carried out and the slots will be returned as they are
        return parsed_output
