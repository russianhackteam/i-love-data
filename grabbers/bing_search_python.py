'''This is a sample code which start search requests to the Bing Image Search API and create a list of url images
Use "pip install azure-cognitiveservices-search-imagesearch" before run the code'''

from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials

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
