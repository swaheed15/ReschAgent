import os
import logging
import streamlit as st
from dotenv import load_dotenv
from scholarly import scholarly
from langchain_community.utilities import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Retrieve the PubMed API Key
pubmed_api_key = os.getenv("PUBMED_API_KEY")
if not pubmed_api_key:
    st.error("PubMed API key is missing. Please add it to the environment variables.")
    st.stop()

# Retrieve the ChatGroq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("ChatGroq API key is missing. Please add it to the environment variables.")
    st.stop()

# Set up PubMed API Wrapper
pubmed_wrapper = PubMedAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300,
    api_key=pubmed_api_key
)

# Initialize PubMedQueryRun tool
pubmed = PubmedQueryRun(api_wrapper=pubmed_wrapper)

# Define Google Scholar Query Function
def google_scholar_query(query, num_results=10, start=0):
    search_results = scholarly.search_pubs(query)
    results = []  # Initialize results as an empty list
    try:
        for _ in range(num_results):
            result = next(search_results)
            results.append(result["bib"])
    except StopIteration:
        return results
    return results

# Wrap Google Scholar query function as a tool
google_scholar_tool = Tool(
    name="GoogleScholarQuery",
    description="Search Google Scholar for academic articles.",
    func=google_scholar_query
)

# Initialize Tools
tools = (pubmed, google_scholar_tool)

# Add sliders for user customization in a compact layout
st.write("Customize your search:")
col1, col2 = st.columns(2)
with col1:
    top_k_results = st.slider(
        "Top Results:",
        min_value=1, max_value=10, value=5, step=1
    )
with col2:
    doc_content_chars_max = st.slider(
        "Max Characters:",
        min_value=100, max_value=500, value=250, step=100
    )

# Display previous messages in a scrollable container
with st.container():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Chat Input Box and Prompt Handling
if prompt := st.chat_input("Search me recent 5 years articles on role of oxytocin in prevention of PPH"):
    if prompt.strip():  # Ensure the input is valid
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Log the slider values for debugging
        logging.info(f"Top K Results: {top_k_results}, Max Characters: {doc_content_chars_max}")

        # Update PubMed Wrapper with slider values
        pubmed_wrapper.top_k_results = top_k_results
        pubmed_wrapper.doc_content_chars_max = doc_content_chars_max

        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            streaming=True
        )

        # Initialize the agent with tools
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True  # Corrected argument
        )

        # Process the query with the agent
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])

            # Append the assistant's response to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            # Display the response
            st.write(response)
    else:
        st.error("Please provide a valid query.")