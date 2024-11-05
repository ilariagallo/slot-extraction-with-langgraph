from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage

PROMPT = """You are a smart travel booking assistant.
    Use the available tools to extract from the message history the necessary information to make a booking.\
    Depending on the booking request, select the relevant tool. \
    Keep asking question to the user until you have filled in all the slots enforced by the tool. \
    Once all slots returned by the tool are filled, ask the user for confirmation of the information you collected. \
    Once they confirmed, tell them they are being transferred to an agent who will help with the booking.
    """


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:

    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("slot_collection", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "slot_collection", False: END}
        )
        graph.add_edge("slot_collection", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result.dict())))
        print("Back to the model!")
        return {'messages': results}
