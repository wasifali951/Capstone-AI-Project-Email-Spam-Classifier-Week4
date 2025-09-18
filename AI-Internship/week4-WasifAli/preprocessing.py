# preprocessing.py
import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs, emails, numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
