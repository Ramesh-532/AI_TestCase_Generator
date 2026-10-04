import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("🧪 AI Test Case Generator from User Stories")
st.caption("Generate structured test cases automatically using Generative AI.")

# Initialize Session State (keeps test cases visible across clicks)
if "test_cases" not in st.session_state:
    st.session_state.test_cases = None

# Sidebar API Key Setup
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # RAG PIPELINE: Local Knowledge Base & Retriever Initialization
    @st.cache_resource
    def setup_rag_retriever():
        loader = TextLoader("qa_standards.txt")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        retriever = TFIDFRetriever.from_documents(chunks)
        retriever.k = 2
        return retriever

    try:
        retriever = setup_rag_retriever()
        st.sidebar.success("QA Knowledge Base Active")
    except Exception as e:
        st.sidebar.error("Knowledge Base Initialization Failed.")
        st.sidebar.caption(f"Details: {e}")
        retriever = None

    # LANGCHAIN PIPELINE: Prompt & Chain Setup
    prompt_template = """
    You are an expert Senior QA Automation Engineer.
    
    Adhere strictly to these retrieved Enterprise QA Standards:
    {context}
    
    User Story / Requirement:
    {user_story}
    
    Output structured test cases in Markdown Table format:
    | Test Case ID | Scenario Name | Type | Pre-conditions | Test Steps | Expected Result | Priority |
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2, google_api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # USER INTERFACE & EXECUTION
    user_story = st.text_area("Enter User Story:", height=150, placeholder="As an online shopper, I want to add an item to my shopping cart...")

    if st.button("Generate Test Cases"):
        if not user_story.strip():
            st.warning("Please enter a requirement.")
        elif retriever is None:
            st.error("Knowledge Base failed to initialize.")
        else:
            with st.spinner("Retrieving QA Standards & Generating Test Cases..."):
                try:
                    retrieved_docs = retriever.invoke(user_story)
                    context_text = format_docs(retrieved_docs)
                    
                    chain = prompt | llm | StrOutputParser()
                    output = chain.invoke({"context": context_text, "user_story": user_story})

                    # Store output persistently in session state
                    st.session_state.test_cases = output
                except Exception as e:
                    st.error(f"Generation Error: {e}")

    # Display test cases and download button persistently if available
    if st.session_state.test_cases:
        st.subheader("Generated Test Suite")
        st.markdown(st.session_state.test_cases)
        
        st.download_button(
            label="Download CSV",
            data=st.session_state.test_cases,
            file_name="test_cases.csv",
            mime="text/csv"
        )
else:
    st.info("Please enter your Gemini API Key in the sidebar to begin.")