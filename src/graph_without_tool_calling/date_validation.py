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


def validate_pick_up_drop_off_dates(parsed_output: Type[Schema]) -> Schema:
    try:
        # Try to parse the date in order to validate them
        pick_up_date = parsed_output.pick_up_date
        drop_off_date = parsed_output.drop_off_date

        if pick_up_date and drop_off_date:
            pick_up_date = datetime.datetime.strptime(pick_up_date, DATE_FORMAT)
            drop_off_date = datetime.datetime.strptime(drop_off_date, DATE_FORMAT)

            if pick_up_date > drop_off_date:
                setattr(parsed_output, 'pick_up_date', "INVALID. Reason: Pick up date after drop off date.")
                setattr(parsed_output, 'drop_off_date', "INVALID. Reason: Pick up date after drop off date.")

        return parsed_output

    except Exception:
        # If parsing fails, validation will not be carried out and the slots will be returned as they are
        return parsed_output
