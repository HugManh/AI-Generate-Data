import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()


def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()


login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

chat_page = st.Page(
    "pages/1_Chat.py", title="Home", icon=":material/chat:", default=True
)
settings_page = st.Page(
    "pages/2_Setting.py", title="Setting", icon=":material/settings:"
)

if st.session_state.logged_in:
    pg = st.navigation(
        [chat_page, settings_page, logout_page]
    )
else:
    pg = st.navigation([login_page])

pg.run()
