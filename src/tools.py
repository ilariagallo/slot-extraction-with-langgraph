import dateparser
import datetime

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.tools import tool
from typing import Optional
from pydantic import Field, BaseModel

from src.azure_chat import model


class BookCar(BaseModel):
    """Information necessary to request a car booking."""

    pick_up_date: Optional[str] = Field(default=None, description="The date of car pick up")
    drop_off_date: Optional[str] = Field(default=None, description="The date of car drop off")
    pick_up_location: Optional[str] = Field(default=None, description="The pick up location for the car rental")
    drop_off_location: Optional[str] = Field(default=None, description="The drop off location for the car rental")
    car_type: Optional[str] = Field(default=None, description="The type of car they want to rent")


@tool
def book_car(user_input: str) -> dict:
    """Returns the information required to book a car in the form of a dictionary."""
    current_date = datetime.datetime.now().date().strftime('%d/%m/%Y')
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value."
                f"When resolving the dates keep in mind today is {current_date} (day/month/year)",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )

    runnable = prompt | model.with_structured_output(schema=BookCar)
    llm_output = runnable.invoke({"text": user_input}).dict()
    llm_output = date_parser(llm_output)
    return llm_output


def date_parser(llm_output: dict) -> datetime:
    for key, value in llm_output.items():
        if 'date' in key and value:
            llm_output[key] = dateparser.parse(value).strftime('%d/%m/%Y')

    return llm_output
