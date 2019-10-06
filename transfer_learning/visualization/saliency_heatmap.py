import cv2

import keras
import numpy as np
from keras.preprocessing import image

from utils import preprocessed_input

_base_sizes = {
    'inceptionv3': (299, 299),
    'vgg16': (224, 224),
    'resnet50': (224, 224)
}


class SaliencyHeatMap:
    def __init__(self):
        self._plotting = {
            'vgg16': 'block5_conv3',
        }

    def get_plot(self, img_path, arch, class_id, base_network=None, target_size=None, layer_name=None):
        original_img = image.img_to_array(image.load_img(img_path))
        small_img = image.img_to_array(image.load_img(img_path, target_size=_base_sizes.get(base_network, target_size)))
        x = np.expand_dims(small_img, axis=0)
        if base_network:
            x = preprocessed_input.get(base_network)(x)

        output = arch.output[:, class_id]
        if not base_network:
            NotImplemented('Not yet implemented with any convnet')
        else:
            if layer_name:
                conv_layer = arch.get_layer(layer_name)
            else:
                if base_network in self._plotting:
                    conv_layer = arch.get_layer(self._plotting.get(base_network))
                else:
                    conv_layer = [l for l in arch.layers if 'conv' in l.name][-1]

            heatmap = self._get_plot(arch, x, output, conv_layer)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = heatmap * 0.4 + original_img
        return heatmap

    @classmethod
    def _get_plot(cls, arch, inp, out, conv_layer):
        grads = keras.backend.gradients(out, conv_layer.output)[0]
        pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))
        iterate = keras.backend.function([arch.input], [pooled_grads, conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([inp])
        for i in range(conv_layer.output_shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        return np.mean(conv_layer_output_value, axis=-1)