import streamlit as st
import requests

st.title("Book Selection Agent")

base_url = "http://127.0.0.1:8000"

genre = st.text_input("Enter a genre")
if st.button("Get Top 100 Books"):
    response = requests.post(f"{base_url}/top-100-books", json={"genre": genre})
    st.write(response.json())

if st.button("Get Top 10 Books"):
    response = requests.get(f"{base_url}/top-10-books")
    st.write(response.json())

book_name = st.text_input("Enter the name of the book you want to select")
if st.button("Select Book"):
    response = requests.post(f"{base_url}/select-book", json={"book_name": book_name})
    st.write(response.json())

if st.button("Conclude"):
    response = requests.get(f"{base_url}/conclude")
    st.write(response.json())
