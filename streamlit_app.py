import streamlit as st

HW1 = st.Page('HW/HW1.py', title='HW1', icon=':material/add_circle:')
HW2 = st.Page('HW/HW2.py', title='HW2', icon=':material/add_circle:')

pg = st.navigation([HW1, HW2])
st.set_page_config(page_title='HW Manager', page_icon=':material/edit:')
pg.run()