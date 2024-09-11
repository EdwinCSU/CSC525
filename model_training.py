import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Function to generate a response from the bot
def generate_response(user_input, chat_history_ids=None):
    # Encode user input and add end-of-sequence token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append user input to the chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate bot response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode bot response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return bot_response, chat_history_ids

# Streamlit interface
def main():
    st.title("CSC525 Chatbot")
    st.write("Type a message to chat with the bot. Type 'quit' to end the conversation.")

    # Initialize chat history
    chat_history_ids = None

    # Text input for user
    user_input = st.text_input("You:", "")

    # When the user submits a message
    if st.button("Send"):
        if user_input.lower() == "quit":
            st.write("Goodbye!")
        else:
            # Generate bot response
            bot_response, chat_history_ids = generate_response(user_input, chat_history_ids)
            st.text_area("Bot:", value=bot_response, height=200, max_chars=None)

if __name__ == "__main__":
    main()

