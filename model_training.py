import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Device setup for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set padding side to left to match the model architecture
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if it is not set

# Move the model to GPU if available
model.to(device)

# Chatbot function to generate response
def generate_response(user_input, chat_history_ids):
    # Encode the user input, add the end-of-sequence token, and move tensors to GPU if available
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    # Append the new user input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the model's response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return bot_response, chat_history_ids

# Streamlit app
def main():
    st.title("Chatbot Interface")
    st.write("Talk to the chatbot below! Type 'quit' to stop the conversation.")

    # Initialize chat history
    chat_history_ids = None

    # Input from user
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if user_input.lower() == "quit":
            st.write("Chat ended. Thank you!")
        else:
            # Generate and display the bot's response
            bot_response, chat_history_ids = generate_response(user_input, chat_history_ids)
            st.text_area("Chatbot:", value=bot_response, height=200, max_chars=None, key=None)

if __name__ == "__main__":
    main()
