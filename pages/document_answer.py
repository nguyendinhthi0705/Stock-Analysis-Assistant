import streamlit as st 
import libs as glib 
from PyPDF2 import PdfReader
import libs as glib 

import base

st.set_page_config(page_title="Hỏi đáp về tài liệu", layout="wide")
st.title('Hỏi đáp về tài liệu')

uploaded_file = st.file_uploader("Tải tài liệu định dạng PDF để hỏi đáp")
docs = []
 

input_text = st.text_input("Câu hỏi của bạn!") 
if uploaded_file is not None and input_text:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        docs.append(page.extract_text())
    
    response = glib.query_document(input_text, docs)
    st.write_stream(response)


   