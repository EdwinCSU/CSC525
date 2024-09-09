from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Device setup for GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set padding side to left to match the model architecture
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if it is not set

# Move the model to GPU if available
model.to(device)

# Chatbot function
def chat_with_bot():
    print("Chat with the bot! Type 'quit' to stop.")
    chat_history_ids = None

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break

        # Encode the new user input, add the end-of-sequence token, and move tensors to GPU if available
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

        # Append the new user input to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        # Generate a response from the model
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode the model's response and print it
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {bot_response}")

# Example usage
if __name__ == "__main__":
    chat_with_bot()