import requests

def collect_conflict_data():
    url = 'https://api.example.com/conflicts'
    response = requests.get(url)
    data = response.json()
    return data
