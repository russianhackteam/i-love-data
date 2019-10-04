from imageai.Detection import ObjectDetection
import os


class ObjectPretrainedDetector:
    __recognized_classes = {'airplane': 'valid',
                            'apple': 'invalid',
                            'backpack': 'invalid',
                            'banana': 'invalid',
                            'baseball_bat': 'invalid',
                            'baseball_glove': 'invalid',
                            'bear': 'invalid',
                            'bed': 'invalid',
                            'bench': 'invalid',
                            'bicycle': 'invalid',
                            'bird': 'invalid',
                            'boat': 'invalid',
                            'book': 'invalid',
                            'bottle': 'invalid',
                            'bowl': 'invalid',
                            'broccoli': 'invalid',
                            'bus': 'invalid',
                            'cake': 'invalid',
                            'car': 'invalid',
                            'carrot': 'invalid',
                            'cat': 'invalid',
                            'cell_phone': 'invalid',
                            'chair': 'invalid',
                            'clock': 'invalid',
                            'couch': 'invalid',
                            'cow': 'invalid',
                            'cup': 'invalid',
                            'dining_table': 'invalid',
                            'dog': 'invalid',
                            'donut': 'invalid',
                            'elephant': 'invalid',
                            'fire_hydrant': 'invalid',
                            'fork': 'invalid',
                            'frisbee': 'invalid',
                            'giraffe': 'invalid',
                            'hair_dryer': 'invalid',
                            'handbag': 'invalid',
                            'horse': 'invalid',
                            'hot_dog': 'invalid',
                            'keyboard': 'invalid',
                            'kite': 'invalid',
                            'knife': 'invalid',
                            'laptop': 'invalid',
                            'microwave': 'invalid',
                            'motorcycle': 'invalid',
                            'mouse': 'invalid',
                            'orange': 'invalid',
                            'oven': 'invalid',
                            'parking_meter': 'invalid',
                            'person': 'invalid',
                            'pizza': 'invalid',
                            'potted_plant': 'invalid',
                            'refrigerator': 'invalid',
                            'remote': 'invalid',
                            'sandwich': 'invalid',
                            'scissors': 'invalid',
                            'sheep': 'invalid',
                            'sink': 'invalid',
                            'skateboard': 'invalid',
                            'skis': 'invalid',
                            'snowboard': 'invalid',
                            'spoon': 'invalid',
                            'sports_ball': 'invalid',
                            'stop_sign': 'invalid',
                            'suitcase': 'invalid',
                            'surfboard': 'invalid',
                            'teddy_bear': 'invalid',
                            'tennis_racket': 'invalid',
                            'tie': 'invalid',
                            'toaster': 'invalid',
                            'toilet': 'invalid',
                            'toothbrush': 'invalid',
                            'traffic_light': 'invalid',
                            'train': 'invalid',
                            'truck': 'invalid',
                            'tv': 'invalid',
                            'umbrella': 'invalid',
                            'vase': 'invalid',
                            'wine_glass': 'invalid',
                            'zebra': 'invalid'
                            }

    def __init__(self, source_dir, output_dir, pretrained_model_name=None, search_for_classes=None,
                 min_proba_to_include=0.3):
        self._source_dir = source_dir
        self._output_dir = output_dir
        self._pretrained_model_name = pretrained_model_name
        self._search_for_classes = search_for_classes
        self._min_proba_to_include = min_proba_to_include * 100
        self._custom_objects = None

        self._detector = ObjectDetection()
        try:
            if pretrained_model_name:
                if pretrained_model_name == 'retina':
                    self._detector.setModelTypeAsRetinaNet()
                    self._detector.setModelPath('./models/coco_model.h5')  # ... retina name
                elif pretrained_model_name == 'yolo':
                    self._detector.setModelTypeAsYOLOv3()
                    self._detector.setModelPath('./models/yolo.h5')  # ... yolo name
                elif pretrained_model_name == 'yolo_small':
                    self._detector.setModelTypeAsTinyYOLOv3()
                    self._detector.setModelPath('./models/yolo-tiny.h5')  # ... yolo small
            else:
                self._detector.setModelTypeAsTinyYOLOv3()
                self._detector.setModelPath('./models/yolo-tiny.h5')  # ... yolo small
        except FileNotFoundError as e:
            raise FileNotFoundError("The model name you provided is not loaded in ./models folder")

        self._detector.loadModel()

        if search_for_classes:
            self._custom_objects = self._detector.CustomObjects(
                **{k: (True if k in search_for_classes else False) for k in
                   ObjectPretrainedDetector.__recognized_classes})
        print('Model loaded!')

    def launch(self):
        imgs = [os.path.join(self._source_dir, x) for x in os.listdir(self._source_dir)]

        for idx, img in enumerate(imgs):
            if self._custom_objects:
                _ = self._detector.detectCustomObjectsFromImage(custom_objects=self._custom_objects,
                                                                         input_image=img,
                                                                         output_image_path=f'{self._output_dir}/{idx}',
                                                                         minimum_percentage_probability=self._min_proba_to_include)
            else:
                _ = self._detector.detectObjectsFromImage(input_image=img,
                                                                   output_image_path=f'{self._output_dir}/{idx}',
                                                                   minimum_percentage_probability=self._min_proba_to_include)