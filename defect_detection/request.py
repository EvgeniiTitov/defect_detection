import requests

REST_API_URl = "http://127.0.0.1:5000/predict"

payload = {
    "pole_number": 12,
    "path_to_data": "D:\Desktop\system_output\FINAL_TILT_TEST"
}


if __name__ == "__main__":

    r = requests.post(url=REST_API_URl,
                      json=payload)

    print(r)

