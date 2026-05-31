# app.py — CPU version for HF Spaces free tier
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from huggingface_hub import login
import torch
import os

st.set_page_config(page_title="MedBot", page_icon="🏥", layout="centered")

st.warning("**Medical Disclaimer:** MedBot is for informational purposes only. Always consult a qualified healthcare provider.", icon="⚠️")
st.title("MedBot — Medical Q&A Assistant")
st.caption("Fine-tuned Google Gemma-2B via QLoRA on 3,000 real patient-doctor conversations")

@st.cache_resource
def load_model():
    token = os.environ.get("HF_TOKEN")
    login(token=token)

    base_model_id = "google/gemma-2b"
    adapter_id = "Abdullah-1-23/medbot-gemma-2b-qlora"

    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def get_response(question):
    prompt = f"""<start_of_turn>user
You are a helpful medical assistant. Answer the patient question clearly and accurately.

Patient: {question}<end_of_turn>
<start_of_turn>model
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

st.divider()
examples = [
    "I have had a persistent cough for two weeks with mild fever.",
    "I am 32 weeks pregnant and experiencing lower back pain.",
    "My child has a rash after eating strawberries.",
]

st.subheader("Try an example")
cols = st.columns(3)
for i, q in enumerate(examples):
    if cols[i].button(q[:35] + "...", key=f"ex_{i}"):
        st.session_state["question"] = q

question = st.text_area(
    "Or type your question:",
    value=st.session_state.get("question", ""),
    height=120,
    placeholder="Describe your symptoms or ask a medical question..."
)

if st.button("Ask MedBot", type="primary", use_container_width=True):
    if question.strip():
        with st.spinner("Analyzing your question..."):
            answer = get_response(question)
        st.subheader("MedBot Response")
        st.markdown(answer)
        st.divider()
        st.caption("Always verify with a licensed physician.")
    else:
        st.error("Please enter a question first.")