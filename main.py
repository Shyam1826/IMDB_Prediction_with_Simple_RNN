import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# 1. Page Configuration
st.set_page_config(
    page_title="IMDb Sentiment Classifier", page_icon="🎬", layout="centered"
)


# 2. Cached Loaders (Prevents reloading on every interaction)
@st.cache_resource
def load_imdb_model():
  # Adjust filename if your saved file is 'simple_rnn_imbd.keras'
  return load_model("simple_rnn_imbd.keras")


@st.cache_data
def get_word_index_data():
  return imdb.get_word_index()


model = load_imdb_model()
word_index = get_word_index_data()


# 3. Preprocessing Helper
def preprocess_text(text, maxlen=200, max_features=10000):
  clean_text = re.sub(r"[^\w\s]", "", text.lower())
  words = clean_text.split()
  encoded_review = [1] + [
      (word_index[word] + 3)
      if (word in word_index and (word_index[word] + 3) < max_features)
      else 2
      for word in words
  ]
  return sequence.pad_sequences(
      [encoded_review], maxlen=maxlen, padding="pre"
  )


# 4. Sidebar Details
with st.sidebar:
  st.header("⚙️ Model Info")
  st.markdown("""
    * **Architecture:** SimpleRNN + Embedding
    * **Vocabulary Size:** 10,000 words
    * **Sequence Length:** 200 tokens
    * **Dataset:** IMDb Reviews
    """)
  st.divider()
  st.caption("Built with TensorFlow & Streamlit")

# 5. Main UI Layout
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type or paste a movie review below to analyze its sentiment.")

# Text Input Area
user_input = st.text_area(
    "Your Review:",
    height=140,
    placeholder="e.g., This movie was fantastic! The acting was great and the plot was thrilling.",
)

# Quick sample buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
  if st.button("👍 Positive Sample"):
    user_input = (
        "An absolute masterpiece! Brilliant direction and stellar"
        " performances throughout."
    )
with col2:
  if st.button("👎 Negative Sample"):
    user_input = (
        "Terrible waste of time. The script was messy and the pacing was"
        " dreadful."
    )

st.write("")

# 6. Classification & Results Display
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
  if not user_input.strip():
    st.warning("⚠️ Please enter a review before analyzing.")
  else:
    with st.spinner("Analyzing text..."):
      processed_input = preprocess_text(user_input, maxlen=200)
      raw_score = float(model.predict(processed_input, verbose=0)[0][0])
      is_positive = raw_score >= 0.5
      confidence = raw_score if is_positive else (1.0 - raw_score)

    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
      if is_positive:
        st.success("### Sentiment: **Positive** 😄")
      else:
        st.error("### Sentiment: **Negative** 😞")

    with res_col2:
      st.metric(label="Model Confidence", value=f"{confidence * 100:.1f}%")

    st.write("**Positivity Meter:**")
    st.progress(raw_score)