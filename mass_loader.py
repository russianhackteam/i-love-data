import os
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', help='theme')
parser.add_argument('-n', '--number', help='number of images')
args = parser.parse_args()

flickr_key = os.getenv("FLICKR_KEY")
flickr_secret = os.getenv("FLICKR_SECRET")
bing_key = os.getenv("BING_KEY")

if args.query is None or args.number is None:
    print("Wrong values! Please use help!")
    sys.exit(1)

if flickr_key is None or flickr_secret is None or bing_key is None:
    print("Please set enviromnent variables!")
    sys.exit(1)


request_to_bing = f"python .//grabbers//bing_downloader.py {args.query} {bing_key} {args.number}"
request_to_duck_duck = f"python .//grabbers//duckduckgo_downloader.py {args.query}"
request_to_flickr = f"python .//grabbers//flickr_downloader.py {args.query} {flickr_key} {flickr_secret}"
request_to_google = f"python .//grabbers//google_downloader.py {args.query} {args.number}"
request_to_yandex = f"python .//grabbers//yandex_downloader.py  Chrome --keywords {args.query} --limit {args.number}"

print(f"Downloading data from Bing!")
subprocess.call(request_to_bing.split())
print(f"Done!")
print(f"Downloading data from Duck Duck Go!")
subprocess.call(request_to_duck_duck)
print(f"Done!")
print(f"Downloading data from Flickr!")
subprocess.call(request_to_flickr)
print(f"Done!")
print(f"Downloading data from Google!")
subprocess.call(request_to_google)
print(f"Done!")
print(f"Downloading data from Yandex!")
subprocess.call(request_to_yandex)
print(f"Done!")
