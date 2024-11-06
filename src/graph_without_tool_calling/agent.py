import datetime

import dateparser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage

from src.graph_without_tool_calling.tools import BookCar


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    slots: dict


class Agent:

    def __init__(self, model, slots, optional_slots_keys, checkpointer):
        graph = StateGraph(AgentState)
        graph.add_node("init_state", self.init_state)
        graph.add_node("slot_collection", self.book_car)
        graph.add_node("conversational_node", self.conversational_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "slot_collection")
        graph.add_edge("slot_collection", "conversational_node")
        graph.add_conditional_edges(
            "slot_collection",
            self.all_slots_collected,
            {True: END, False: "conversational_node"}
        )
        self.graph = graph.compile(checkpointer=checkpointer)
        self.model = model
        self.slots = slots
        self.optional_slots_keys = optional_slots_keys

    def init_state(self, state: AgentState):
        state['slots'] = self.slots
        return state

    def book_car(self, state: AgentState):
        """Extracts necessary information to book a car from the text and message history."""
        user_input = state['messages'][-1].content
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert extraction algorithm. "
                    "Extract relevant information from the text and message history. "
                    "If you do not know the value of an attribute asked to extract, "
                    "return null for the attribute's value."
                    f"Don't try to resolve dates yourself. Populate the attribute's value with what the user provided."
                    "Message history:"
                    "{messages}",
                ),
                # Please see the how-to about improving performance with
                # reference examples.
                # MessagesPlaceholder('examples'),
                ("human", "{text}"),
            ]
        ).partial(messages=state.get("messages", []))

        runnable = prompt | self.model.with_structured_output(schema=BookCar)
        llm_output = runnable.invoke({"text": user_input}).dict()
        parsed_output = self.date_parser(llm_output)
        output = self.validate_pick_up_drop_off_dates(parsed_output)

        state['slots'].update({key: value for key, value in output.items() if value})
        return {'messages': state['messages'], 'slots': state['slots']}

    @staticmethod
    def date_parser(llm_output: dict) -> datetime:
        for key, value in llm_output.items():
            if 'date' in key and value:
                parsed_date = dateparser.parse(value)
                llm_output[key] = parsed_date.strftime('%d/%m/%Y') if parsed_date else value

        return llm_output

    @staticmethod
    def validate_pick_up_drop_off_dates(parsed_output: dict):
        try:
            # Try to parse the date in order to validate them
            pick_up_date = parsed_output['pick_up_date']
            drop_off_date = parsed_output['drop_off_date']

            if pick_up_date and drop_off_date:
                pick_up_date = datetime.datetime.strptime(pick_up_date, '%d/%m/%Y')
                drop_off_date = datetime.datetime.strptime(drop_off_date, '%d/%m/%Y')

                if pick_up_date > drop_off_date:
                    parsed_output['pick_up_date'] = "INVALID. Reason: Pick up date after drop off date."
                    parsed_output['drop_off_date'] = "INVALID. Reason: Pick up date after drop off date."

            return parsed_output

        except Exception:
            # If parsing fails, validation will not be carried out and the slots will be returned as they are
            return parsed_output

    def all_slots_collected(self, state: AgentState):
        slots = state['slots']
        if None in slots.values():
            return False
        else:
            return True

    def conversational_node(self, state: AgentState):
        user_input = state['messages'][-1].content

        mandatory_slots = {k: v for k, v in state.get("slots", {}).items() if k not in self.optional_slots_keys}
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a smart travel booking assistant. "
                    "You are helping a customer collecting the information required to proceed with their booking."
                    "You have to ask questions to the users in oder to fill in all the missing values in the Slots."
                    "Only ask questions related to the slots provided. No other question allowed."
                    "If a slot is set to INVALID ask the user for clarification and tell them the reason why."
                    "Once all the slots are filled, ask the user to confirm the details provided and then tell them "
                    "they are being route to an agent who will help with making the booking."
                    "Message history:"
                    "{messages}"
                    "Slots:"
                    "{slots}",
                ),
                # Please see the how-to about improving performance with
                # reference examples.
                # MessagesPlaceholder('examples'),
                ("human", "{text}"),
            ]
        ).partial(messages=state.get("messages", []), slots=mandatory_slots)

        chain = prompt | self.model
        ai_message = chain.invoke({'text': user_input})
        return {'messages': [ai_message], 'slots': state['slots']}
