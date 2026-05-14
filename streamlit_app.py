import streamlit as st
import pandas as pd
from transformers import pipeline

st.success("Gratulację! Z powodzeniem uruchomiłeś aplikację.")

st.title("LAB04: Streamlit")

st.header("Wprowadzenie do zajęć")

st.subheader("O streamlit")

st.text("To przykładowa aplikacja z wykorzystaniem Streamlit")

# podobne do text ale lepsza manipulacja tekstem
st.write("Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego")

st.code("st.wtite()", language="python")

with st.echo():
    st.write("Echo")

df = pd.read_csv("DSP_4.csv", sep=";")
st.dataframe(df)

st.header("Przetwarzanie języka naturalnego")

# huggingface - fajna stronka z bazami

option = st.selectbox(
    "Opcje",
    {
        "Wydźwięk emocjonalny tekstu (ang)",
        "???" # do domu
    },
)
if option == "Wydźwięk emocjonalny tekstu (ang)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
