import streamlit as st 
import libs as glib 
import json

st.set_page_config(page_title="Home")

input_text = st.chat_input()
if input_text: 
    response = glib.call_claude_sonet_stream(input_text)
    st.write_stream(response)

    



    
   