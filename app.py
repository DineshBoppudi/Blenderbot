from flask import Flask, render_template, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load model
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    inputs = tokenizer([user_message], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
