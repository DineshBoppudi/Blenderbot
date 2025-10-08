from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

print("🤖 BlenderBot Chatbot Ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Chatbot: Goodbye 👋")
        break

    # Encode input
    inputs = tokenizer([user_input], return_tensors="pt")

    # Generate response safely
    reply_ids = model.generate(**inputs, max_length=100)
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    print(f"Chatbot: {bot_reply}")
