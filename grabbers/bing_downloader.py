
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials
import urllib.request
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import URLError, HTTPError
from urllib.parse import quote
import http.client
from http.client import IncompleteRead, BadStatusLine

DOWNLOADS_DIR='downloads'

def search(query,key,count):
    client = ImageSearchAPI(CognitiveServicesCredentials(key))
    image_results = client.images.search(query=query, count=count)
    urls = []

    for i in range(len(image_results.value)):
        image = image_results.value[i]
        urls.append(image.content_url)
    
    for link in urls:
        link = link.strip()
        name = link.rsplit('/', 1)[-1]
        filename = os.path.join(DOWNLOADS_DIR, name)
        #print(link)
        if not os.path.exists(DOWNLOADS_DIR):
            os.makedirs(DOWNLOADS_DIR)
        if not os.path.isfile(filename):
            print('Downloading: ' + filename)
            try:
                urllib.request.urlretrieve(link, filename)
            except Exception as inst:
                print(inst)
                print('  Encountered unknown error. Continuing.')

def main(query,key,count):
    search(query,key,count) 
    # do whatever the script does

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])