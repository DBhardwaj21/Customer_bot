import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize

url = 'https://www.paisaintime.com/repay'
response = requests.get(url)


if response.status_code == 200:
    
    soup = BeautifulSoup(response.content, 'html.parser')

    
    text = soup.get_text(separator='\n', strip=True)
    with open('extracted_textrepay.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    
    # print(text)
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    
