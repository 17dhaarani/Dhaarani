import os
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCStewsCoyboaNEHaHoeYBetjUePREagyU"  # Replace with your actual key

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Search tool
search_tool = DuckDuckGoSearchRun()

# Tool list
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_tool.run,
        description="Use this to search the web for factual and current info."
    )
]

# Memory for chat
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent only once
if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    st.session_state.chat_history = []

# Streamlit UI
st.title("🤖 Gemini + DuckDuckGo Chatbot")

user_input = st.text_input("Ask me anything:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent.run(user_input)
            st.session_state.chat_history.append((user_input, response))
        except Exception as e:
            response = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append((user_input, response))

# Show conversation history
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
