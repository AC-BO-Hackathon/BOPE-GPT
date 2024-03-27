import requests

data = {'value': 'Hello from external script!'}
response = requests.post('http://localhost:8080/submit', json=data)

print(response.text)
