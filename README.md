# i-love-data
## General approach on collecting custom image datasets for classification purpouses

1.  ### Search for prepared datasets:
* [Kaggle dataset search](https://www.kaggle.com/datasets)
* [CoCo Dataset](http://cocodataset.org/#explore)
* [ImageNet dataset](http://www.image-net.org/)
* [Google dataset search](https://toolbox.google.com/datasetsearch)

2.  ### Search for prepared datasets:
* Get youtube video and cut it on frames
* Use image search API 
* [Azure cognitive services (Bing Web Search API)](https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/)
* [DuckDuckGo (DuckDuckGo Search API)](https://github.com/deepanprabhu/duckduckgo-images-api)

3.  ### Awesome links:
* [Pyimagesearch](hhttps://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/) blog about image processing and machine learning 

* https://towardsdatascience.com/creating-a-dataset-using-an-api-with-python-dcc1607616d

4.  ### Tool for object detections:
* Classical problem is when you can google for some pictures of your interest (like "can of cola") and the picture you find contains several objects (different cans). So you need to somehow preprocess source image and extract regions of your interest. This is where pretrained YOLO or Retina may be useful (or your custom last trained dense layer using transfer learning). Implementation can be found [here](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md)

* Instead of using the raw library we had written a simple wrapper that accepts directories with pictures and target classes you want to extract. Assuming you want to extract cans from some folder with pictures:


![cans](https://i.ibb.co/Z2XGmjb/inp-img.jpg)

After we apply wrapper to the source image we can get seaparate cans and after it feed it directly to the network.
![box](https://i.ibb.co/7gqxrV3/0.png)


5. ### Annotation:
* [labelme](https://github.com/wkentaro/labelme) is a super simple python module with gui for any annotation task: classification, segmentation, bb-boxes and more. 
