import requests

url = 'http://localhost:5000/predict'
files = {'image': ('Rodwiler.jpeg', open('Rodwiler.jpeg', 'rb'), 'image/jpeg')}
#files = {'image': ('Schnauzer-miniatura-3-800x780.jpg', open('Schnauzer-miniatura-3-800x780.jpg', 'rb'), 'image/jpeg')}
response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print('Error al procesar la imagen')