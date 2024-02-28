import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

LOGGER = st.logger

def generate_text(prompt, max_length=50, num_return_sequences=3, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature)
    generated_texts = [tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in output]
    return generated_texts

def run():
    st.set_page_config(
        page_title="Text Generation with Hugging Face",
        page_icon="🤖",
    )

    st.write("# Text Generation with Hugging Face Model 🤖")

    st.sidebar.info("Enter your prompt on the left and click 'Generate'.")

    prompt = st.text_input("Enter your prompt:", "How to train a machine learning model?")

    if st.button("Generate"):
        generated_texts = generate_text(prompt)
        for i, text in enumerate(generated_texts):
            st.write(f"Generated text {i+1}: {text}")

if __name__ == "__main__":
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    run()
