import requests
import json

REST_API_URl = "http://127.0.0.1:5000/predict"

# payloads = [
#     {
#         "pole_number": 1,
#         "path_to_data": r"D:\Desktop\system_output\INPUT\1"
#     },
#     {
#         "pole_number": 4,
#         "path_to_data": r"D:\Desktop\system_output\INPUT\4"
#     },
#     {
#         "pole_number": 9,
#         "path_to_data": r"D:\Desktop\system_output\INPUT\9"
#     },
#     {
#         "pole_number": 228,
#         "path_to_data": r"D:\Desktop\system_output\INPUT\228"
#     }
# ]

payloads = [
    {
        "pole_number": 1,
        "path_to_data": r"D:\Desktop\system_output\INPUT\1"
    },
    {
        "pole_number": 4,
        "path_to_data": r"D:\Desktop\system_output\INPUT\4"
    }
]


if __name__ == "__main__":

    for payload in payloads:
        r = requests.post(url=REST_API_URl,
                          json=payload)

        print(json.dumps(r.json(), indent=4))
        print("\n"*3)
