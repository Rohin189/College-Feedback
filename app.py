from flask import Flask, render_template, request
from transformers import pipeline
import pandas as pd
import os

app = Flask(__name__)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Academics", "Facilities", "Administration"]

CSV_FILE = "classified_feedback.csv"

@app.route("/", methods=["GET"])
def home():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        history = df.to_dict(orient="records")
    else:
        history = []

    return render_template("form.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    feedback = request.form["feedback"]

    result = classifier(feedback, candidate_labels)
    top_label = result["labels"][0]
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["feedback", "predicted_category"])

    new_entry = {"feedback": feedback, "predicted_category": top_label}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    history = df.to_dict(orient="records")

    return render_template("form.html", prediction=top_label, history=history)

if __name__ == "__main__":
    app.run(debug=True)