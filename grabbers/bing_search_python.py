'''This is a sample code which start search requests to the Bing Image Search API and create a list of url images
Use "pip install azure-cognitiveservices-search-imagesearch" before run the code'''

from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials
import urllib.request
from urllib.request import Request, urlopen
from urllib.request import URLError, HTTPError
from urllib.parse import quote
import http.client
from http.client import IncompleteRead, BadStatusLine

'''This function download your image from url to your downloads folder'''

def download_image(image_url):
    main_directory = "downloads"
    extensions = (".jpg", ".gif", ".png", ".bmp", ".svg", ".webp", ".ico")
    url = image_url
    try:
        os.makedirs(main_directory)
    except OSError as e:
        if e.errno != 17:
            raise
        pass
    req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
    response = urlopen(req, None, 10)
    data = response.read()
    response.close()
    image_name = str(url[(url.rfind('/')) + 1:])
    if '?' in image_name:
        image_name = image_name[:image_name.find('?')]
    if any(map(lambda extension: extension in image_name, extensions)):
        file_name = main_directory + "/" + image_name
    else:
        file_name = main_directory + "/" + image_name + ".jpg"
        image_name = image_name + ".jpg"
    try:
        output_file = open(file_name, 'wb')
        output_file.write(data)
        output_file.close()
    except IOError as e:
        raise e
    except OSError as e:
        raise e
    print("completed ====> " + image_name.encode('raw_unicode_escape').decode('utf-8'))
    
'''Instead YOUR_SUBSCRIPTION_KEY write your subcription key on azure and also instead
YOUR_THEME_OF_SEARCH write your request to Bing for the images you want to find

Examples:

subscription_key = "3a125g7050324108ab3f4b9bc3ls24k9"
search_term = "Katy Perry"'''

subscription_key = "YOUR_SUBSCRIPTION_KEY"
search_term = "YOUR_THEME_OF_SEARCH"

client = ImageSearchAPI(CognitiveServicesCredentials(subscription_key))

'''By default cognitive services find 35 images. You can change this number using the count parameter.
And specifically in this code - changing the values of the variable your_count'''

your_count = 150

image_results = client.images.search(query=search_term, count=your_count)

'''If you write right subscription_key, code below will write you amount of images that it find and the first image that he find'''

if image_results.value:
    first_image_result = image_results.value[0]
    print("Total number of images returned: {}".format(len(image_results.value)))
    print("First image thumbnail url: {}".format(
        first_image_result.thumbnail_url))
    print("First image content url: {}".format(first_image_result.content_url))
else:
    print("No image results returned!")

'''This is how you call every images from the list'''

some_image = image_results.value[1]
print(some_image.content_url)

'''Download all images'''

for i in range(len(image_results.value)):
    image = image_results.value[i]
    download_image(image.content_url)
