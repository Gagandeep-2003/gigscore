import requests
import json

try:
    print("Testing GET /prompt...")
    res = requests.get("https://text.pollinations.ai/what is the capital of france")
    print(res.status_code, res.text[:100])
except Exception as e:
    print("GET error", e)

try:
    print("Testing POST / (OpenAI compatible)...")
    res = requests.post(
        "https://text.pollinations.ai/",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "openai"
        }
    )
    print(res.status_code, res.text[:100])
except Exception as e:
    print("POST error", e)

try:
    print("Testing POST /v1/chat/completions ...")
    res = requests.post(
        "https://gen.pollinations.ai/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "openai"
        }
    )
    print(res.status_code, res.text[:100])
except Exception as e:
    print("POST error", e)
