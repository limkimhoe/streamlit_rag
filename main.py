import streamlit as st
from helper_functions.utility import check_password

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="AI Chatbot",
    page_icon="🤖",
)

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

# endregion <--------- Streamlit Page Configuration --------->

st.title("🤖 AI Chatbot")
st.write("""
Welcome to the AI Chatbot! Navigate using the **sidebar** to explore:

- 💬 **Chatbot** — Have a conversation with the AI assistant
- ℹ️ **About** — Learn more about this app
""")
