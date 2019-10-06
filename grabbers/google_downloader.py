from google_images_download import google_images_download
import sys

def search(query,limit):
    response = google_images_download.googleimagesdownload()   
    arguments = {"keywords":query,"limit":limit,"print_urls":True}   
    paths = response.download(arguments)  
    print(paths)

def main(query,limit):
    search(query,limit) 

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
