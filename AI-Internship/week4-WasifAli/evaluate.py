# evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from preprocessing import clean_text
from sklearn.model_selection import train_test_split

def load_data(path="sms.tsv"):
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'text'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def evaluate(pipeline_path="models/spam_pipeline.joblib"):
    df = load_data()
    X = df['clean_text']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = joblib.load(pipeline_path)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham","spam"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Spam Classifier")
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.show()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Spam Classifier")
    plt.legend()
    plt.savefig("roc_curve.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    evaluate()
