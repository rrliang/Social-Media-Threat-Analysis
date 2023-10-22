# Class:        ImageCaption.py
# Description:  Image captioning module
# Author:       A.Cedeno, R. Liang
# Ver:          11/11/22

import os
import pickle
import numpy as np
from pathlib import Path

from nltk.translate.bleu_score import corpus_bleu
from tqdm.notebook import tqdm
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
# from nltk.translate.bleu_score import corpus_bleu

class ImageCaption:

    def __init__(self):
        BASE_DIR = '../resources/flickr8k'

    def create_model(self):
        # BASE_DIR = '../resources/flickr8k'
        # WORKING_DIR = Path('../outputs/image')

        # load vgg16 model
        model = VGG16()
        # restructure the model
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # summarize
        print(model.summary())

        # folder path
        # dir_path = r'/content/drive/MyDrive/combined1600'
        # count = 0
        # # Iterate directory
        # for path in os.listdir(dir_path):
        #     # check if current path is a file
        #     if os.path.isfile(os.path.join(dir_path, path)):
        #         count += 1
        # print('File count:', count)

        # extract features from image
        features = {}
        directory = Path('../resources/test-images')

        for img_name in os.listdir(directory):
            # load the image from file
            img_path = Path('../resources/test-images/' + img_name)
            image = load_img(img_path, target_size=(224, 224))
            # convert image pixels to numpy array
            image = img_to_array(image)
            # reshape data for model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # preprocess image for vgg
            image = preprocess_input(image)
            # extract features
            feature = model.predict(image, verbose=0)
            # get image ID
            image_id = img_name.split('.')[0]
            # store feature
            features[image_id] = feature

        # # store features in pickle
        # pickle.dump(features, open(Path('../models/image/features.pkl'), 'wb'))

        with open(Path('../resources/test-img-captions.txt'), 'r') as f:
            next(f)
            captions_doc = f.read()

        # create mapping of image to captions
        mapping = {}
        # process lines
        for line in tqdm(captions_doc.split('\n')):
            # split the line by comma(,)
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            # remove extension from image ID
            image_id = image_id.split('.')[0]
            # convert caption list to string
            caption = " ".join(caption)
            # create list if needed
            if image_id not in mapping:
                mapping[image_id] = []
            # store the caption
            mapping[image_id].append(caption)

        def clean(mapping):
            for key, captions in mapping.items():
                for i in range(len(captions)):
                    # take one caption at a time
                    caption = captions[i]
                    # preprocessing steps
                    # convert to lowercase
                    caption = caption.lower()
                    # delete digits, special chars, etc.,
                    caption = caption.replace('[^A-Za-z]', '')
                    # delete additional spaces
                    caption = caption.replace('\s+', ' ')
                    # add start and end tags to the caption
                    caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
                    captions[i] = caption

        # before preprocess of text
        print(mapping['gun-270_jpgrf906483df35f030ed11b943f4bec27aa9'])

        # preprocess the text
        clean(mapping)

        # after preprocess of text
        print(mapping['gun-270_jpgrf906483df35f030ed11b943f4bec27aa9'])

        all_captions = []
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)

        # tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1

        # get maximum length of the caption available
        max_length = max(len(caption.split()) for caption in all_captions)

        image_ids = list(mapping.keys())
        split = int(len(image_ids) * 0.90)
        train = image_ids[:split]
        test = image_ids[split:]

        # create data generator to get data in batch (avoids session crash)
        def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
            # loop over images
            X1, X2, y = list(), list(), list()
            n = 0
            while 1:
                for key in data_keys:
                    n += 1
                    captions = mapping[key]
                    # process each caption
                    for caption in captions:
                        # encode the sequence
                        seq = tokenizer.texts_to_sequences([caption])[0]
                        # split the sequence into X, y pairs
                        for i in range(1, len(seq)):
                            # split into input and output pairs
                            in_seq, out_seq = seq[:i], seq[i]
                            # pad input sequence
                            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                            # encode output sequence
                            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                            # store the sequences
                            X1.append(features[key][0])
                            X2.append(in_seq)
                            y.append(out_seq)
                    if n == batch_size:
                        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                        yield [X1, X2], y
                        X1, X2, y = list(), list(), list()
                        n = 0

        # encoder model
        # image feature layers
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.4)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        # sequence feature layers
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256)(se2)

        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)

        import tensorflow as tf

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        METRICS = [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)

        # plot the model
        plot_model(model, show_shapes=True)

        # save the model
        # model.save(Path('../outputs/image/best_model.h5'))

        # train the model
        epochs = 20
        batch_size = 32
        steps = len(train) // batch_size

        for i in range(epochs):
            # create data generator
            generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
            # fit for one epoch
            model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1, batch_size=batch_size)

        def idx_to_word(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None

        # generate caption for an image
        def predict_caption(model, image, tokenizer, max_length):
            # add start tag for generation process
            in_text = 'startseq'
            # iterate over the max length of sequence
            for i in range(max_length):
                # encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad the sequence
                sequence = pad_sequences([sequence], max_length)
                # predict next word
                yhat = model.predict([image, sequence], verbose=0)
                # get index with high probability
                yhat = np.argmax(yhat)
                # convert index to word
                word = idx_to_word(yhat, tokenizer)
                # stop if word not found
                if word is None:
                    break
                # append word as input for generating next word
                in_text += " " + word
                # stop if we reach end tag
                if word == 'endseq':
                    break

            return in_text


        # validate with test data
        actual, predicted = list(), list()

        for key in tqdm(test):
            # get actual caption
            captions = mapping[key]
            # predict the caption for image
            y_pred = predict_caption(model, features[key], tokenizer, max_length)
            # split into words
            actual_captions = [caption.split() for caption in captions]
            y_pred = y_pred.split()
            # append to the list
            actual.append(actual_captions)
            predicted.append(y_pred)

        # calcuate BLEU score
        print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

        from PIL import Image
        import matplotlib.pyplot as plt

        def generate_caption(img_name):
            # load the image
            # image_name = "1001773457_577c3a7d70.jpg"
            image_id = img_name.split('.')[0]
            # img_path = os.path.join("/content/", image_name)
            # img_path = os.path.join(BASE_DIR, "Images", image_name)
            img_path = Path('../resources/test-images/' + img_name)
            image = Image.open(img_path)
            captions = mapping[image_id]
            print('---------------------Actual---------------------')
            for caption in captions:
                print(caption)
            # predict the caption
            y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
            print('--------------------Predicted--------------------')
            print(y_pred)
            plt.imshow(image)

        # generate_caption('knife-88_jpgrf3eb00f50e4524d8cfd555915016531ad.jpg')
        with open(Path('../models/image/tokenizer.pickle'), 'wb') as f:
            pickle.dump(tokenizer, f)
        print(model.metrics_names)
        print(features['knife-88_jpgrf3eb00f50e4524d8cfd555915016531ad'])
        # save the model
        model.save(Path('../models/image/best_model_reduced2.h5'))
        # print(predict_caption(model, ))

    def idx_to_word(self, integer, tokenizer):
        # print("tokenizer word index " + str(tokenizer.word_index.items()))
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(self, model, image, tokenizer, max_length):
        # add start tag for generation process
        in_text = 'startseq'
        # iterate over the max length of sequence
        for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length)
            # predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = self.idx_to_word(yhat, tokenizer)

            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                break

        return in_text

    def extract_features(self, filename):
        # load the model
        model = VGG16()
        # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # load the photo
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        return feature

    def run_model(self, image_path):
        from keras.models import load_model
        import tensorflow
        ###### CHANGE THE H5 MODEL YOU WANT TO RUN BELOW
        model = tensorflow.keras.models.load_model(Path('../models/image/best_model_reduced.h5'), compile=False)
        # model = load_model(Path('../models/image/best_model_2.h5'))

        # summarize model.
        # model.summary()

        with open(Path('../models/image/tokenizer.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)

        image = self.extract_features(image_path)
        predicted_caption = self.predict_caption(model, image, tokenizer, 29)

        # Remove startseq and endseq
        query = predicted_caption
        stopwords = ['startseq', 'endseq']
        querywords = query.split()
        resultwords = [word for word in querywords if word.lower() not in stopwords]
        result = ' '.join(resultwords)
        return result
