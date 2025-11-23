#to run locally - 
# pip install gradio
# python spam_mail_app.py
# Visit → http://127.0.0.1:7860

import gradio as gr
import pickle


# Load your saved model and vectorizer (after you pickle them in your notebook)
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_spam(text):
    if not text.strip():
        return "Please enter an email message."
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return "SPAM" if prediction == 1 else "HAM (Safe Email)"

app = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=6, placeholder="Paste email text here..."),
    outputs="text",
    title="Spam Mail Classifier",
    description="Detects whether an email is Spam or Ham."
)

app.launch()
