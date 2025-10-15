import requests

r = requests.post("http://127.0.0.1:8000/search/text", data={"query": "red dress", "k": 5})
print(r.json())

with open(r"D:\path\to\image.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/search/image", files={"file": f}, data={"k": 5})
    print(r.json())