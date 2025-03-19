import streamlit as st
from utils.gemini_helper import GeminiAnalyzer
import asyncio

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

async def get_ai_response(analyzer, question):
    try:
        response = await analyzer.analyze_data(question)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("AI Data Analysis Chat")
    
    if not st.session_state.get('data_cleaned', False):
        st.error(" Please clean your data first in the Data Cleaning page!")
        st.stop()
        return
    
    if 'cleaned_data' not in st.session_state:
        st.error(" No cleaned data found. Please process your data in the Data Cleaning page!")
        st.stop()
        return
    
    df = st.session_state['cleaned_data']

    initialize_chat_history()

    gemini_analyzer = GeminiAnalyzer(df)

    with st.expander(" View Data Preview", expanded=False):
        st.dataframe(df.head())
        st.write(f"Total Records: {len(df)}")
        st.write(f"Columns: {', '.join(df.columns)}")

    st.markdown("###  Chat with Your Data")


    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.write(content)
    

    if prompt := st.chat_input("Ask a question about your data..."):

        with st.chat_message("user"):
            st.write(prompt)
        add_message("user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = asyncio.run(get_ai_response(gemini_analyzer, prompt))
                st.write(response)
        add_message("assistant", response)

    if st.sidebar.button(" Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    
    if st.sidebar.button(" Download Chat History"):
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        ])
        st.sidebar.download_button(
            label="Save Chat",
            data=chat_text,
            file_name="data_analysis_chat.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main() 