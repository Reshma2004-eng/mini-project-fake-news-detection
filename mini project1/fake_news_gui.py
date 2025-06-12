import tkinter as tk
from tkinter import messagebox
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)         # remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r'\d+', '', text)           # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text

# Prediction function
def predict_news():
    input_text = text_entry.get("1.0", tk.END).strip()
    if input_text == "":
        messagebox.showwarning("Input Error", "Please enter some news text.")
        return

    cleaned = clean_text(input_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    result_label.config(text=f"Prediction: {prediction.upper()} news", fg="green" if prediction == "real" else "red")

# Clear function
def clear_input():
    text_entry.delete("1.0", tk.END)
    result_label.config(text="Prediction: ", fg="black")

# GUI setup
window = tk.Tk()
window.title("Fake News Detection")
window.geometry("500x400")
window.config(bg="#f0f0f0")

tk.Label(window, text="Enter News Text:", font=("Arial", 12), bg="#f0f0f0").pack(pady=10)
text_entry = tk.Text(window, height=10, width=60, font=("Arial", 10))
text_entry.pack()

tk.Button(window, text="Predict", command=predict_news, font=("Arial", 12), bg="#4caf50", fg="white", width=15).pack(pady=10)
tk.Button(window, text="Clear", command=clear_input, font=("Arial", 12), bg="#f44336", fg="white", width=15).pack()

result_label = tk.Label(window, text="Prediction: ", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=20)

window.mainloop()

