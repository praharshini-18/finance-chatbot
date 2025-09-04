from flask import Flask, request, render_template_string, redirect, url_for
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load API token
load_dotenv()
app = Flask(__name__)

# Initialize Hugging Face Inference Client
client = InferenceClient(
    model="mistralai/Mistral-Nemo-Instruct-2407",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Store chat history in memory
chat_history = []

@app.route("/", methods=["GET"])
def home():
    chat_html = ""
    for user, bot in chat_history:
        chat_html += f"""
        <div class='chat user'><b>You:</b> {user}</div>
        <div class='chat bot'><b>Bot:</b> {bot}</div>
        """

    return render_template_string(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>💰 Finance Chatbot</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f1f1f1;
                margin: 0;
                height: 100vh;
            }}
            .chat-container {{
                display: flex;
                flex-direction: column;
                width: 100%;
                height: 100vh;
                background: white;
            }}
            .chat-header {{
                background: #2ecc71;
                color: white;
                padding: 15px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                flex-shrink: 0;
            }}
            .chat-box {{
                flex: 1;
                padding: 10px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }}
            .chat {{
                margin: 8px 0;
                padding: 10px;
                border-radius: 10px;
                max-width: 75%;
                word-wrap: break-word;
            }}
            .user {{
                background: #d1f7c4;
                align-self: flex-end;
            }}
            .bot {{
                background: #e0e0e0;
                align-self: flex-start;
            }}
            .chat-input {{
                display: flex;
                border-top: 1px solid #ccc;
                flex-shrink: 0;
            }}
            .chat-input input {{
                flex: 1;
                padding: 10px;
                border: none;
                outline: none;
                font-size: 16px;
            }}
            .chat-input button {{
                padding: 10px 15px;
                background: #2ecc71;
                color: white;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }}
            .chat-input button:hover {{
                background: #27ae60;
            }}
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">💬 Finance Chatbot</div>
            <div class="chat-box" id="chat-box">{chat_html}</div>
            <form action="/chat" method="post" class="chat-input">
                <input type="text" name="message" placeholder="Type your message..." required>
                <button type="submit">Send</button>
            </form>
        </div>
        <script>
            // Auto-scroll to bottom on new message
            var chatBox = document.getElementById("chat-box");
            chatBox.scrollTop = chatBox.scrollHeight;
        </script>
    </body>
    </html>
    """)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]

    # Call Hugging Face chat model
    response = client.chat.completions.create(
        model="mistralai/Mistral-Nemo-Instruct-2407",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=200
    )

    bot_reply = response.choices[0].message["content"]

    # Save to history
    chat_history.append((user_message, bot_reply))

    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
