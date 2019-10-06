import cv2
import pickle
import random
import string

import numpy as np
from keras.optimizers import RMSprop, Adam
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import pandas as pd
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

from utils import preprocessed_input, base_sizes, base_network_init, base_output_pooling
from visualization.saliency_heatmap import SaliencyHeatMap

_explainer = SaliencyHeatMap()


class KerasLearner:
    def __init__(self, layers=None, dropout=None, base='inceptionv3', pooling='average', objective='binary', **kwargs):
        if dropout is None:
            dropout = [0.5]
        if layers is None:
            layers = [512]
        if len(layers) != len(dropout):
            raise ValueError('Expected to see dropout and layers array of the same shape')
        self._layers = layers
        self._dropout = dropout
        self._base = base
        self._pooling = pooling
        self._compiled_model = None
        self._objective = objective
        self._classes = kwargs.get('classes') if 'classes' in kwargs else None
        self._checkpoint_path = '{}_{}.h5'.format(self._base,
                                                  ''.join(random.choice(string.ascii_lowercase) for _ in range(10)))
        if 'path' in kwargs:
            self._compile(kwargs.get('path'))

        self._class_mode = 'binary' if self._objective == 'binary' else 'categorical'

        self._class_to_idx = None
        self._idx_to_class = None

    def fit_from_frame(self, train, val, x_col, y_col, train_augs_generator=None,
                       val_augs_generator=None, epochs=10, batch_size=32, unfreeze=0, optimizer=None,
                       use_default_callbacks=True, custom_callbacks=None,
                       load_from_checkpoint=False):
        if not self._classes:
            self._classes = train[y_col].unique().tolist()
            self._class_to_idx = {value: index for index, value in enumerate(self._classes)}
            self._idx_to_class = {index: value for index, value in enumerate(self._classes)}

        if custom_callbacks is None:
            custom_callbacks = []

        if not train_augs_generator:
            train_datagen = ImageDataGenerator(preprocessing_function=preprocessed_input.get(self._base),
                                               rotation_range=30,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')

            train_augs_generator = train_datagen.flow_from_dataframe(train,
                                                                     x_col=x_col,
                                                                     y_col=y_col,
                                                                     target_size=base_sizes.get(self._base),
                                                                     batch_size=batch_size,
                                                                     classes=self._classes,
                                                                     class_mode=self._class_mode)

        if not val_augs_generator:
            val_datagen = ImageDataGenerator(preprocessing_function=preprocessed_input.get(self._base))
            val_augs_generator = val_datagen.flow_from_dataframe(val,
                                                                 shuffle=False,
                                                                 x_col=x_col,
                                                                 y_col=y_col,
                                                                 target_size=base_sizes.get(self._base),
                                                                 batch_size=batch_size,
                                                                 classes=self._classes,
                                                                 class_mode=self._class_mode)

        train_steps = train.shape[0] // batch_size
        val_steps = val.shape[0] // batch_size

        if use_default_callbacks:
            custom_callbacks = custom_callbacks + self._default_callbacks()
        path = self._checkpoint_path if load_from_checkpoint else None
        self._compile(path=path, optimizer=optimizer, unfreeze=unfreeze)

        return self._compiled_model.fit_generator(
            train_augs_generator,
            steps_per_epoch=train_steps,
            validation_data=val_augs_generator,
            validation_steps=val_steps,
            callbacks=custom_callbacks,
            epochs=epochs,
            verbose=0)

    def predict_from_frame(self, frame, x_col):
        if not self._compiled_model:
            ValueError('Model is not fitted yet')
        datagen = ImageDataGenerator(preprocessing_function=preprocessed_input.get(self._base))
        batch_generator = datagen.flow_from_dataframe(frame,
                                                      shuffle=False,
                                                      x_col=x_col,
                                                      y_col=None,
                                                      target_size=base_sizes.get(self._base),
                                                      batch_size=1,
                                                      class_mode=None)
        return self._compiled_model.predict_generator(batch_generator, steps=frame.shape[0])

    def predict_from_file(self, img_name, convert_to_classes=False, threshold=0.5):
        pil_img = image.load_img(img_name, target_size=base_sizes.get(self._base))
        img_as_array = image.img_to_array(pil_img)
        preprocessed = preprocessed_input.get(self._base)(img_as_array)
        img_tensor = np.expand_dims(preprocessed, axis=0)

        prediction = self._compiled_model.predict(img_tensor)[0]
        if convert_to_classes:
            if len(self._classes) == 2:
                prediction[prediction > threshold] = 1
                return np.array([self._classes[int(x)] for x in prediction])
            else:
                prediction = np.array([self._classes[int(x.argmax())] for x in prediction])
        return prediction

    def plot_precision_recall_curve(self, frame, x_col, y_col):
        if len(self._classes) != 2:
            IndexError('Precision Recall curve is only available for binary tasks')

        predictions = self.predict_from_frame(frame, x_col)
        targets = np.array([self._class_to_idx[x] for x in frame[y_col].values])
        precision, recall, _ = precision_recall_curve(targets, predictions)
        plt.step(recall, precision, color='r', alpha=0.4, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        steps = np.linspace(0, 1, 11)
        plt.xticks(steps), plt.yticks(steps)
        plt.xlabel('Recall'), plt.ylabel('Precision')
        plt.ylim([0.0, 1.05]), plt.xlim([0.0, 1.0])

    def explain_prediction(self, img_name, layer_name=None, threshold=0.5, filename='explanation.bmp',
                           return_as_thumbnail=True):
        prediction = self.predict_from_file(img_name, threshold=threshold)
        if len(prediction) == 1:
            if prediction[0] < threshold:
                print('Predicted class {}, no explanation provided'.format(self._classes[0]))
                return self._classes[0], None
            print('Predicted class {}, output heatmap'.format(self._classes[1]))
            class_name = self._classes[1]
            heatmap = _explainer.get_plot(img_name, self._compiled_model, 0, layer_name=layer_name,
                                          base_network=self._base)
        else:
            class_id = np.argmax(prediction)
            class_name = self._classes[int(class_id)]
            print('Predicted class {}, output heatmap for a given class'.format(class_name))
            heatmap = _explainer.get_plot(img_name, self._compiled_model, class_id, layer_name=layer_name,
                                          base_network=self._base)
        if filename:
            cv2.imwrite(filename, heatmap)
            print('Explanation saved to {}'.format(filename))
        pil = image.array_to_img(heatmap)
        if return_as_thumbnail:
            pil.thumbnail(base_sizes.get(self._base))
        return class_name, pil

    def save(self, filename):
        if not self._compiled_model:
            ValueError('Model is not compiled')
        fields = {k[1:] if k[0] == '_' else k: v for k, v in self.__dict__.items()
                  if k not in ['_compiled_model']}
        self._compiled_model.save_weights(self._checkpoint_path)
        fields['path'] = self._checkpoint_path
        with open(filename, 'wb') as f:
            pickle.dump(fields, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            fields = pickle.load(f)

        return KerasLearner(**fields)

    def get_arch(self):
        return self._compiled_model

    def get_classes(self):
        return self._classes

    def compile(self, path=None, optimizer=None, unfreeze=0):
        return self._compile(path, optimizer, unfreeze)

    def _compile(self, path=None, optimizer=None, unfreeze=0):
        if self._base not in base_network_init:
            ValueError('Unknown base network, available are: {}'.format(','.join(base_network_init.keys())))
        if self._pooling not in base_output_pooling:
            ValueError('Unknown pooling layer, available are: {}'.format(','.join(base_network_init.keys())))

        base = base_network_init.get(self._base)(weights='imagenet', include_top=False)
        base.trainable = False
        x = base.output
        x = base_output_pooling.get(self._pooling)()(x)
        for dense_units, dropout_units in zip(self._layers, self._dropout):
            x = keras.layers.Dense(dense_units)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dropout(rate=dropout_units)(x)
        activation = 'sigmoid' if self._objective == 'binary' else 'softmax'
        output_units = 1 if self._objective == 'binary' else len(self._classes)
        output = keras.layers.Dense(output_units, activation=activation)(x)

        self._compiled_model = keras.Model(inputs=base.input, outputs=output)
        optimizer = optimizer if optimizer else keras.optimizers.Adam()

        if path:
            self._compiled_model.load_weights(path)
            print('Weights loaded')
        if unfreeze > 0:
            for layer in self._compiled_model.layers[:len(base.layers)]:
                layer.trainable = False
            for layer in self._compiled_model.layers[::-1]:
                if unfreeze <= 0:
                    break
                if 'conv2d' in layer.name:
                    unfreeze -= 1
                layer.trainable = True

        self._compiled_model.compile(
            loss='binary_crossentropy' if self._objective == 'binary' else 'categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        print('Model compiled')

    def _default_callbacks(self):
        return [
            ModelCheckpoint(self._checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.65, min_lr=0.00001),
            EarlyStopping(monitor='val_loss', patience=8),
            TQDMNotebookCallback()]

if __name__ == '__main__':
    frame = pd.read_csv('labeled_data.csv')
    train, val = train_test_split(data, stratify=frame.target, test_size=0.2, shuffle=True, random_state=555)

    size = (299, 299)
    batch_size = 32
    optimizer = Adam()

    learner = KerasLearner(objective='binary', base='inceptionv3', layers=[512, 256], dropout=[0.5, 0.5])
    history = learner.fit_from_frame(train, val, x_col='image_path', y_col='target', optimizer=optimizer, epochs=10)

    learner.save('inception_base.pickle')

    # learner.predict_from_file(img_name='./test_img.png')
