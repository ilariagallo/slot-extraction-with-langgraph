from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage

from src.graph_without_tool_calling.date_validation import date_parser, validate_pick_up_drop_off_dates


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    slots: BaseModel


class Agent:

    def __init__(self, model, slots, optional_slots_keys, checkpointer):
        graph = StateGraph(AgentState)
        graph.add_node("init_state", self.init_state)
        graph.add_node("slot_collection", self.collect_slots)
        graph.add_node("conversational_node", self.conversational_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "slot_collection")
        graph.add_edge("slot_collection", "conversational_node")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.model = model
        self.slots = slots
        self.optional_slots_keys = optional_slots_keys

    def init_state(self, state: AgentState):
        state['slots'] = self.slots
        return state

    def collect_slots(self, state: AgentState):
        """Extract relevant information from the text and message history."""
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

        runnable = prompt | self.model.with_structured_output(schema=self.slots.__class__)
        llm_output: BaseModel = runnable.invoke({"text": user_input})
        parsed_output: BaseModel = date_parser(llm_output)
        output: BaseModel = validate_pick_up_drop_off_dates(parsed_output)

        slots = state['slots'].copy(update={key: value for key, value in output.dict().items() if value})
        return {'messages': state['messages'], 'slots': slots}

    def conversational_node(self, state: AgentState):
        user_input = state['messages'][-1].content

        mandatory_slots = {k: v for k, v in state.get("slots").dict().items() if k not in self.optional_slots_keys}
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
        ).partial(messages=state.get("messages"), slots=mandatory_slots)

        chain = prompt | self.model
        ai_message = chain.invoke({'text': user_input})
        return {'messages': [ai_message], 'slots': state['slots']}
