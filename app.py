import google.generativeai as genai
import streamlit as st

# Page Config
st.set_page_config(page_title="AI Test Case Generator", layout="wide")

st.title("🧪 AI Test Case Generator from User Stories")
st.caption("Generate structured test cases automatically using Generative AI.")

# Session State Initialization (prevents download button from disappearing on click)
if "test_cases" not in st.session_state:
    st.session_state.test_cases = None
if "active_model" not in st.session_state:
    st.session_state.active_model = None

# Sidebar - API Key Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    st.markdown("[Get Google Gemini API Key](https://aistudio.google.com/)")

# Input Section
user_story = st.text_area(
    "Enter User Story / Requirement:",
    height=150,
    placeholder="Example: As a registered user, I want to log in using my email and password so that I can access my account dashboard.",
)

# Execution Button
if st.button("Generate Test Cases", type="primary"):
    if not api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar!")
    elif not user_story.strip():
        st.warning("Please enter a user story first.")
    else:
        try:
            genai.configure(api_key=api_key)

            # Fetch available content generation models
            available_models = [
                m.name
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]

            if not available_models:
                st.error(
                    "No content generation models found for this API Key."
                )
            else:
                prompt = f"""
                You are an expert QA Quality Assurance Automation Engineer.
                Convert the following User Story into complete, structured Test Cases.

                User Story:
                {user_story}

                Output Requirement:
                For each test scenario (Positive, Negative, and Edge Cases), provide:
                1. Test Case ID & Title
                2. Preconditions / Conditions
                3. Detailed Test Steps (Numbered)
                4. Expected Result

                Ensure maximum coverage (80%+ target) and clean markdown formatting.
                """

                # Filter out deprecated models (like 2.5) to prevent 404 errors
                candidate_models = [
                    m for m in available_models if "2.5" not in m
                ]
                if not candidate_models:
                    candidate_models = available_models

                response = None
                successful_model = None

                with st.spinner("AI Agent is generating test cases..."):
                    # Fallback loop: tries models sequentially until generation succeeds
                    for model_name in candidate_models:
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(prompt)
                            successful_model = model_name
                            break
                        except Exception:
                            continue

                if response and response.text:
                    st.session_state.test_cases = response.text
                    st.session_state.active_model = successful_model
                else:
                    st.error(
                        "Unable to generate response. Please verify your API key and permissions."
                    )

        except Exception as e:
            st.error(f"Error generating test cases: {str(e)}")

# Display Results & Persistent Download Button
if st.session_state.test_cases:
    st.info(f"Connected using model: `{st.session_state.active_model}`")
    st.success("Test Cases Generated Successfully!")
    st.markdown(st.session_state.test_cases)

    st.divider()

    st.download_button(
        label="📥 Download Test Cases (.md)",
        data=st.session_state.test_cases,
        file_name="generated_test_cases.md",
        mime="text/markdown",
        use_container_width=True,
    )