import streamlit as st
from transformers import pipeline

# Load Model (runs once)

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

generator = load_model()


# Streamlit UI

st.set_page_config(page_title="IdeaScribe AI", layout="centered")

st.title("🤖 IdeaScribe AI")
st.write("Enter a prompt and generate text")

# User Input
prompt = st.text_area("Enter your prompt:", "Once upon a time")

max_length = st.slider("Max Length", min_value=20, max_value=200, value=50)

temperature = st.slider("Creativity (Temperature)", 0.5, 1.5, 1.0)

# Generate Button
if st.button("Generate Text"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating..."):
            result = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1
            )

            st.subheader("Generated Text:")
            st.write(result[0]['generated_text'])
