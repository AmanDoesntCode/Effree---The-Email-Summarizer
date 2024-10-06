import streamlit as st

# Sidebar for navigation
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Go to", ["Home", "Another Page"])

if page == "Home":
    from HomePage import EmailApp
    email_app = EmailApp()
    email_app.run()

elif page == "Another Page":
    st.title("Another Page")
    st.write("This is another page for different functionality.")
