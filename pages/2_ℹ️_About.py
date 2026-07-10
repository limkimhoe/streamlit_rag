import streamlit as st
from helper_functions.utility import check_password

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="About",
    page_icon="ℹ️",
)


# endregion <--------- Streamlit Page Configuration --------->

st.title("ℹ️ About This App")

st.write("""
This is an AI Chatbot built with **Streamlit** and **OpenAI**.

### Features
- 💬 **Conversational AI** — Chat naturally with GPT-4o-mini
- 🔒 **Password Protected** — Secure access with a shared password
- 🔄 **Streaming Responses** — See the AI's reply as it's generated
- 📊 **Token Tracking** — Monitor estimated token usage in the sidebar

### Tech Stack
| Component | Tool |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io/) |
| Language Model | [OpenAI GPT-4o-mini](https://openai.com/) |
| Token Counting | [tiktoken](https://github.com/openai/tiktoken) |
| Backend | Python 3.10+ |

### Disclaimer
*This application is for educational and demonstration purposes. The information provided by the AI should not be taken as professional advice.*
""")
