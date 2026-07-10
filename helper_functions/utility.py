import streamlit as st
import hmac


def check_password(
    secret_key: str = "password",
    session_key: str = "password_correct",
    input_key: str = "password",
) -> bool:
    """Returns `True` if the user entered the correct password.

    Args:
        secret_key:  Key in st.secrets that holds the expected password.
        session_key: Session-state key used to persist auth status.
        input_key:   Streamlit widget key for the text_input (must be
                     unique per page to avoid session-state collisions).
    """

    def password_entered():
        if hmac.compare_digest(
            st.session_state[input_key], st.secrets[secret_key]
        ):
            st.session_state[session_key] = True
            del st.session_state[input_key]
        else:
            st.session_state[session_key] = False

    if st.session_state.get(session_key, False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key=input_key
    )
    if session_key in st.session_state:
        st.error("😕 Password incorrect")
    return False
