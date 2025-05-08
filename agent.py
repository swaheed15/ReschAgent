import streamlit as st
import logging
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun  # Corrected import
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
from scholarly import scholarly
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Retrieve the PubMed API Key
pubmed_api_key = os.getenv("PUBMED_API_KEY")

# Ensure the API key is available for PubMed
if not pubmed_api_key:
    raise ValueError("PubMed API key is missing. Please add it to the environment variables.")

# Set up PubMed API Wrapper with the API key
pubmed_wrapper = PubMedAPIWrapper(top_k_results=1, doc_content_chars_max=300, api_key=pubmed_api_key)

# Initialize PubMedQueryRun tool with PubMed API Wrapper
pubmed = PubmedQueryRun(api_wrapper=pubmed_wrapper)

# Google Scholar Query Function using scholarly
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

# Wrap google_scholar_query for LangChain compatibility
google_scholar_tool = Tool(
    name="GoogleScholarQuery",
    description="Search Google Scholar for academic articles.",
    func=google_scholar_query
)

# Add sliders for user customization in the sidebar
st.sidebar.title("Customize your search:")
top_k_results = st.sidebar.slider(
    "Select the number of the top results:",
    min_value=1, max_value=10, value=5, step=1
)
doc_content_chars_max = st.sidebar.slider(
    "Select the maximum number of characters for each document summary:",
    min_value=100, max_value=500, value=250, step=100
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Input Box and Prompt Handling
if prompt := st.chat_input("Search me recent 5 years articles on role of oxytocin in prevention of PPH"):
    if prompt.strip():  # Ensure the input is valid
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Log the slider values for debugging
        logging.info(f"Top K Results: {top_k_results}, Max Characters: {doc_content_chars_max}")

        # Pass the values to PubMed API Wrapper or other tools
        pubmed_wrapper.top_k_results = top_k_results
        pubmed_wrapper.doc_content_chars_max = doc_content_chars_max

        # Set up the callback handler for Streamlit messages
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Initialize the agent with PubMed and Google Scholar tools
        tools = (pubmed, google_scholar_tool)

        llm = ChatGroq(
            groq_api_key=pubmed_api_key,
            model_name="gemma2-9b-it",
            streaming=True
        )

        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handling_parsing_error=True
        )

        with st.chat_message("assistant"):
            # Run the agent with the current messages
            response = search_agent.run(st.session_state.messages, callback=[st_cb])

            # Append the assistant's response to the session messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            # Display the response from the assistant
            st.write(response)
    else:
        st.error("Please provide a valid query.")

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
