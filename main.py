import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
from llm_helper import get_few_shot_db_chain
import streamlit as st


st.title("Atharv Inventry: Database Q&A")

question=st.text_input("Question:")

if question:
    chain=get_few_shot_db_chain()
    answer=chain.run(question)
    st.write(answer)

   
    
