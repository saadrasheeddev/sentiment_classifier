import streamlit as st
from streamlit_chat import message
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.title("Sentiment Classifier")
@st.cache_resource(show_spinner=True)
def load_model_tokenizer():
    peft_model_id = "lora-flan-t5-large-sentiment"
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=256).input_ids.to("cpu")
    outputs = model.generate(input_ids=input_ids, top_p=0.9, max_length=256)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

message("Hi I am Sentiment Analysis Classifier. Drop a Sentiment here.", is_user=False)

placeholder = st.empty()
placeholder_2 = st.empty()

prompt = """
        Human: I absolutely love my SR Watch! The design is sleek, the battery life is impressive, and it syncs seamlessly with my smartphone. It has truly enhanced my daily routine.
        Assistant: Positive

        Human: My experience with the SR Watch has been disappointing. The battery drains quickly, and the interface is clunky. I expected better performance considering the price.
        Assistant: Negative

        Human: The SR Watch has some good features, but there are also some drawbacks. The design is modern, but the battery life could be better. Overall, it's an okay product.
        Assistant: Neutral
    """

input_ = st.text_input("Human", key="input_field")

if st.button("Generate"):
    with placeholder.container():
        message(input_, is_user=True)
    input_ = prompt + "\nHuman: " + input_ + ". Assistant: "
    with st.spinner(text="Classifying Review.....  "):
        with placeholder_2.container():
            message(inference(model, tokenizer, input_), is_user=False)
