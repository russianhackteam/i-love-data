import flickr_api
import urllib
import os
import sys
import random 

DOWNLOADS_DIR = 'downloads'

def get_url(text,key,secret):
    print ('start')
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

def main(query, key, secret):
    for link in get_url(query,key,secret):
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

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
    