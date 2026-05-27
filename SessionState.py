"""Small compatibility wrapper around Streamlit session state."""

import streamlit as st


class SessionState:
    """Attribute-style access to ``st.session_state``."""

    def __getattr__(self, name):
        try:
            return st.session_state[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        st.session_state[name] = value


def get(**kwargs):
    """Initializes missing keys and returns an attribute-style state object."""
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value
    return SessionState()
