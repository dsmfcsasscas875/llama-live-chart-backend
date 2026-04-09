import requests

def test_register():
    url = "http://127.0.0.1:8000/api/v1/register"
    data = {
        "email": "test_" + str(int(__import__('time').time())) + "@example.com",
        "password": "testpassword123"
    }
    try:
        response = requests.post(url, json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_register()
