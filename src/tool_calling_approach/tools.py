import dateparser
import datetime

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.tools import tool
from typing import Optional
from pydantic import Field, BaseModel

from src.tool_calling_approach.agent import AgentState
from src.tool_calling_approach.azure_chat import model


class BookCar(BaseModel):
    """Information necessary to request a car booking.
    Please stick to these fields when asking questions to the user."""

    pick_up_location: Optional[str] = Field(default=None, description="The pick up location for the car rental")
    pick_up_date: Optional[str] = Field(default=None, description="Pick up date")
    pick_up_time: Optional[str] = Field(default=None, description="Pick up time")
    drop_off_location: Optional[str] = Field(default=None, description="The drop off location for the car rental")
    drop_off_date: Optional[str] = Field(default=None, description="Drop off date")
    drop_off_time: Optional[str] = Field(default=None, description="Drop off time")
    car_type: Optional[str] = Field(default=None, description="The type of car the user wants to rent")


@tool
def book_car(user_input: str, state: AgentState) -> BookCar:
    """This tool helps with extracting from the text the necessary information to book a car.
    This tool will also help you format the information collected from the user.
    Please make sure to run it every time the user provides new information together with context from the
    message history. This will serve as a summary of all information collected so far."""

    current_date = datetime.datetime.now().date().strftime('%d/%m/%Y')
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Extract relevant information from the text and message history. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value."
                f"When resolving the dates keep in mind today is {current_date} (day/month/year)"
                "Message history:"
                "{messages}",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    ).partial(messages=state.get("messages", []))

    runnable = prompt | model.with_structured_output(schema=BookCar)
    llm_output = runnable.invoke({"text": user_input}).dict()
    llm_output = date_parser(llm_output)
    return BookCar.model_validate(llm_output)


def date_parser(llm_output: dict) -> datetime:
    for key, value in llm_output.items():
        if 'date' in key and value:
            llm_output[key] = dateparser.parse(value).strftime('%d/%m/%Y')

    return llm_output
