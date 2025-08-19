from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# ---- STATE ----
class ConversationState(Dict[str, Any]):
    pass

# ---- STM Agent ----
def stm_agent(state: ConversationState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    user_message = state.get("user_message", "")
    last_message = state.get("last_message", "")

    prompt = f"""
    You are the STM Agent.
    The last message was: {last_message}
    Current user message: {user_message}
    Respond concisely.
    """

    response = llm.invoke(prompt)

    # update STM (only last message is kept)
    state["last_message"] = user_message
    state["response"] = response.content
    return state

# ---- LTM Agent ----
def ltm_agent(state: ConversationState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    user_message = state.get("user_message", "")
    memory_context = state.get("memory_context", "")

    prompt = f"""
    You are the LTM Agent.
    Stored facts: {memory_context}
    Current user message: {user_message}
    Respond concisely.
    """

    response = llm.invoke(prompt)

    # update LTM if user says "remember"
    if "remember" in user_message.lower():
        state["memory_context"] += f"\n{user_message}"

    state["response"] = response.content
    return state

# ---- Controller ----
def controller(state: ConversationState):
    user_message = state.get("user_message", "")
    if "remember" in user_message.lower() or "recall" in user_message.lower():
        state["route"] = "ltm"
    else:
        state["route"] = "stm"
    return state

# ---- Build Graph ----
def build_graph():
    workflow = StateGraph(ConversationState)

    workflow.add_node("controller", controller)
    workflow.add_node("stm", stm_agent)
    workflow.add_node("ltm", ltm_agent)

    workflow.set_entry_point("controller")

    workflow.add_conditional_edges(
        "controller",
        lambda state: state["route"],
        {"stm": "stm", "ltm": "ltm"},
    )

    workflow.add_edge("stm", END)
    workflow.add_edge("ltm", END)

    return workflow

graph = build_graph()

# memory backends
checkpointer = MemorySaver()                     # STM (per session)
store = InMemoryStore(namespace="ltm-multi")     # LTM (persists facts)

# compiled graph
app = graph.compile(checkpointer=checkpointer, store=store)
