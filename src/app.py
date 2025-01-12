import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
import operator
import os

# Define the Analyst model
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

# Define the ResearchGraphState
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str

# Define the builder for the state graph
def create_builder():
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", conduct_interview)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    return builder

# Placeholder functions for the nodes
def create_analysts(state: ResearchGraphState):
    # Logic to create analysts
    return state

def human_feedback(state: ResearchGraphState):
    # Logic to handle human feedback
    return state

def conduct_interview(state: ResearchGraphState):
    # Logic to conduct interviews
    return state

def write_report(state: ResearchGraphState):
    # Logic to write the report
    return state

def write_introduction(state: ResearchGraphState):
    # Logic to write the introduction
    return state

def write_conclusion(state: ResearchGraphState):
    # Logic to write the conclusion
    return state

def finalize_report(state: ResearchGraphState):
    # Logic to finalize the report
    return state

def initiate_all_interviews(state: ResearchGraphState):
    # Logic to initiate all interviews
    return state

def main():
    st.title("Research Assistant with Streamlit")

    # Input fields for API keys
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    tavily_api_key = st.text_input("Enter your Tavily API Key:", type="password")

    # Input fields for research topic and number of analysts
    topic = st.text_input("Enter the research topic:")
    max_analysts = st.number_input("Enter the number of analysts:", min_value=1, value=3)

    # Additional field to add another analyst
    additional_analyst = st.text_input("Add another analyst (optional):")

    # Button to start the research
    if st.button("Start Research"):
        if not openai_api_key or not tavily_api_key:
            st.error("Please enter both OpenAI and Tavily API keys.")
            return

        # Set environment variables for API keys
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Create analysts
        analysts = [
            Analyst(
                affiliation="Interactive Design Studio",
                name="Sophie Lin",
                role="Interaction Designer",
                description="Sophie designs user interfaces and experiences for AI systems in smart cities. Her focus is on creating intuitive and accessible interactions between citizens and AI technologies, ensuring that these systems are user-friendly and enhance the quality of urban life."
            )
        ]

        if additional_analyst:
            analysts.append(Analyst(
                affiliation="Custom Affiliation",
                name="Custom Analyst",
                role="Custom Role",
                description=additional_analyst
            ))

        # Initialize the state
        state = ResearchGraphState(
            topic=topic,
            max_analysts=max_analysts,
            human_analyst_feedback="",
            analysts=analysts,
            sections=[],
            introduction="",
            content="",
            conclusion="",
            final_report=""
        )

        # Create the builder and compile the graph
        builder = create_builder()
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        # Define a unique thread_id for the graph execution
        thread_id = "research_thread_1"

        # Run the research graph with the configurable dictionary
        result = graph.invoke(state, {"configurable": {"thread_id": thread_id}})

        # Display the final report
        st.markdown("### Final Report")
        st.markdown(result["final_report"])
