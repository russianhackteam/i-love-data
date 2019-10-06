from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Polar bears","limit":5,"print_urls":True, "no_download":True}   #creating list of arguments
#paths = response.download(arguments)   #passing the arguments to the function
links = response.download(arguments)
links = str(links)
start = 0
end = 0
for i in range(len(links)):
    if (links[i] == '['):
        start = i
    if (links[i] == ']'):
        end = i

start += 2
end += 1

links = links[start:end]

urls = links.split("', '")

print(urls[2])
