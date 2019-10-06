import flickr_api
import os
import random 

key = ''
secret = ''

def get_url(text):
    
    flickr_api.set_keys(api_key=key, api_secret=secret)

    w = flickr_api.Walker(
        flickr_api.Photo.search, 
        text=text, 
        license='2,3,4,5,6,9', 
        media="photos",
        orientation="",
        sort='interestingness-desc',
        safe_search=3
    )
    photos = [next(w) for _ in range(124)]
    photo = random.choice(photos)
    links_to_photo = []
    for photo in photos:
        links_to_photo.append(photo.getPhotoFile())
    return links_to_photo

if __name__ == '__main__':
    print(get_url("hack"))
