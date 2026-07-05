import requests

def collect_news_data():
    url = 'https://api.example.com/news'
    response = requests.get(url)
    data = response.json()
    return data
