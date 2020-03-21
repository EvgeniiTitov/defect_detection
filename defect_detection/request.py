import requests

REST_API_URl = "http://127.0.0.1:5000/predict"

payload = {
    "pole_number": 15,
    "path_to_data": r"D:\Desktop\system_output\TEST_concrete_only"
}


if __name__ == "__main__":

    r = requests.post(url=REST_API_URl,
                      json=payload)

    print(r.json())

# JSON with video works bad

# Try several requests here in a FOR loop and see how it handles it:
# an image, a video and a folder.

# Server freezes after one request

# Thorough tests