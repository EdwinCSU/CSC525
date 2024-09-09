import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose a smaller model for Streamlit Cloud
model_name = "distilgpt2"  # You can experiment with other smaller models

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Set padding side to left to match the model architecture
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Chatbot function
def generate_response(user_input, chat_history_ids):
  new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
  bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)   
 if chat_history_ids is not None else new_user_input_ids
  chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)   

  bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)   

  return bot_response, chat_history_ids

def main():
  st.title("Chatbot Interface")
  st.write("Interact with the chatbot! Type 'quit' to end the chat.")

  chat_history_ids = None

  user_input = st.text_input("You:", "")

  if st.button("Send"):
    if user_input:
      bot_response, chat_history_ids = generate_response(user_input, chat_history_ids)
      st.text_area("Chatbot:", value=bot_response, height=200, max_chars=None, key=None)

if __name__ == "__main__":
  main()
