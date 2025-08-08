import streamlit as st
from model_utils import load_embedded_chunks, find_relevant_chunks, build_prompt, generate_answer_llama_cpp

st.set_page_config(page_title="Policy Document Q&A Assistant")
st.title("Policy Document Q&A Assistant")

# Load embedded chunks once
@st.cache_resource
def load_chunks():
    return load_embedded_chunks()

embedded_chunks = load_chunks()

query = st.text_input("Ask a question about the policy document:")
if query:
    relevant_chunks = find_relevant_chunks(query, embedded_chunks)
    prompt = build_prompt(query, relevant_chunks)
    answer = generate_answer_llama_cpp(prompt)

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("Context used:"):
        for chunk in relevant_chunks:
            st.markdown(f"*Page {chunk['page_number']}* - {chunk['text']}")