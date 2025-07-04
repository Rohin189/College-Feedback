import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

def classify_feedback():
    # Load feedback data
    df = pd.read_csv("feedback.csv")

    # Create zero-shot classifier pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Define your categories
    candidate_labels = ["Academics", "Facilities", "Administration"]

    # Classify each feedback entry
    predicted_labels = []
    for feedback in df['feedback']:
        result = classifier(feedback, candidate_labels)
        predicted_labels.append(result['labels'][0])  # take top predicted label

    # Add predictions to dataframe
    df['predicted_category'] = predicted_labels

    # Save to new CSV
    df.to_csv("classified_feedback.csv", index=False)

    print("Classification completed! Check classified_feedback.csv")

    counts = df['predicted_category'].value_counts()
    counts.plot(kind='bar')
    plt.title('Feedback Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Number of Feedbacks')
    plt.show()

if __name__ == "__main__":
    classify_feedback()