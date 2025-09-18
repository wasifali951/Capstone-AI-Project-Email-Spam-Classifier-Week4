# data_fetch.py
import os
import requests

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
OUT = "sms.tsv"

def download():
    if os.path.exists(OUT):
        print(f"{OUT} already exists.")
        return
    r = requests.get(URL)
    r.raise_for_status()
    with open(OUT, "wb") as f:
        f.write(r.content)
    print("Downloaded sms dataset as sms.tsv")

if __name__ == "__main__":
    download()
