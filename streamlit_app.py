import streamlit as st
import pandas as pd
import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import st.iframe as components

st.title("Aplikacja NLP z Streamlit")

st.write("---")

st.header("O aplikacji")
st.write("Ta aplikacja wykorzystuje modele HuggingFace do analizy wydźwięku emocjonalnego oraz tłumaczenia tekstu z języka angielskiego na niemiecki. Te demo używa M2M100 Translator od firmy Meta (facebook/m2m100_1.2B model).")

tenor_embed_code = """
<div class="tenor-gif-embed" data-postid="20844378" data-share-method="host" data-aspect-ratio="1.77857" data-width="100%">
    <a href="https://tenor.com/view/hack-gif-20844378">Hack GIF</a>
</div>
<script type="text/javascript" async src="https://tenor.com/embed.js"></script>
"""

components.html(tenor_embed_code, height=400)

st.header("Jak używać aplikacji?")
st.write("Wybierz jedną z opcji z listy rozwijanej.")

st.subheader("Opcja *'Wydźwięk emocjonalny tekstu (ang)'*")
st.write("Po wpisaniu tekstu naciśnij kombinację ***Ctrl+Enter*** lub przycisk ***Analizuj*** aby uzyskać wynik który powie o wydźwięku danego tekstu i jego procent zgodności.")

st.subheader("Opcja *'Tłumaczenie (ang -> niem)'*")
st.write("Po wpisaniu tekstu naciśnij kombinację ***Ctrl+Enter*** lub przycisk ***Tłumacz*** aby zobaczyć wynik tłumaczenia tekstu z angielskiego na niemiecki.")

@st.cache_resource
def load_translator():
    try:
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Nie udało się załadować modelu: {e}")
        return None, None


st.write("---")

st.header("Przetwarzanie języka naturalnego")
st.subheader("Wybierz opcję z menu poniżej")

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (ang)",
        "Tłumaczenie (ang -> niem)"
    ]
)

if option == "Wydźwięk emocjonalny tekstu (ang)":
    from transformers import pipeline
    text = st.text_area(label="Wpisz tekst po angielsku", key="text_sentiment")
    if st.button("Analizuj"):
        if text:
            with st.spinner("Analizuję..."):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
                label = answer[0]['label']
                score = round(answer[0]['score'], 4) * 100
                st.success(f"Wydźwięk: {label}; Zgodność: {score}%;")
        else:
            st.warning("Wpisz tekst do analizy.")

elif option == "Tłumaczenie (ang -> niem)":
    text = st.text_area(label="Wpisz tekst do przetłumaczenia na niemiecki", key="text_translation")
    if st.button("Przetłumacz"):
        if text:
            with st.spinner("Tłumaczenie w toku..."):
                try:
                    tokenizer, model = load_translator()
                    if tokenizer is None or model is None:
                        st.error("Model nie jest dostępny. Spróbuj odświeżyć stronę.")
                    else:
                        tokenizer.src_lang = "en"
                        encoded = tokenizer(text, return_tensors="pt")

                        with torch.no_grad():
                            generated_tokens = model.generate(
                                **encoded,
                                forced_bos_token_id=tokenizer.get_lang_id("de")
                            )

                        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                        st.success(translated)
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas tłumaczenia: {e}")
        else:
            st.warning("Wpisz tekst do przetłumaczenia.")


st.markdown("<p style='text-align: center; color: gray;'>Autor: s28889 | LAB04 | 2026</p>", unsafe_allow_html=True)