# Class:        TextClassification.py
# Description:  Uses NLP to classify text as either concerning or non-concerning
# Author:       R. Liang
# Ver:          11/9/22

import os
import pickle
import re
from pathlib import Path
from random import randrange

import matplotlib.patches as matplotlib_patches
import pandas as pd
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from spacy.lang.en import STOP_WORDS
from wordcloud import WordCloud
import tensorflow_hub as hub


class TextClassification:

    def __init__(self):
        self._path_to_dataset = Path('../resources/text-post-dataset.csv')
        self._contractions = {}
        self._col_name = ['Sentiment', 'Text']  # Label all the columns
        self._pandas_dataset = pd.read_csv(self._path_to_dataset, encoding='latin', header=None,
                                           names=self._col_name)  # Read csv
        self._pandas_dataset = self._pandas_dataset.sample(
            frac=1)  # Sample all rows. frac=1 means return all rows in random order
        self._dataset_size = len(self._pandas_dataset)
        self._pandas_dataset = self._pandas_dataset[:self._dataset_size]  # Out of samplesize it takes a sample.

        self._x = 0
        self._y = 0
        self._x_train = 0
        self._x_test = 0
        self._y_train = 0
        self._y_test = 0
        self._x_test_tf = 0
        self._x_train_tf = 0
        self._x_tf = 0
        self._tf = 0

    @staticmethod
    def check_dir():
        folders = [Path('../outputs/pickled'), Path('../outputs/roc'), Path('../outputs/accuracy'),
                   Path('../outputs/lime')]
        for i in range(len(folders)):
            if not os.path.isdir(folders[i]):
                os.makedirs(folders[i])

    def cont_to_exp(self, text):
        if type(text) is str:
            for key in self._contractions:
                value = self._contractions[key]
                text = text.replace(key, value)
            return text
        else:
            return text

    def preprocess_dataset(self):
        path_to_contractions = Path('../resources/preprocessing-contractions.txt')

        # Open contraction list and add to the contraction dictionary
        # List from https://www.kaggle.com/raymant/text-processing-sentiment-analysis?scriptVersionId=33503187&cellId=23
        contractions_file = open(path_to_contractions, 'r')

        for line in contractions_file:
            split = line.strip('\n').split(':')
            self._contractions.update({split[0]: split[1]})

        self._pandas_dataset['Text'] = self._pandas_dataset['Text'].astype('str')
        # Remove links with https
        self._pandas_dataset.Text = self._pandas_dataset.Text.apply(lambda x: re.sub(r'http?:\/\/\S+', '', x))
        # Remove links with www. and com
        self._pandas_dataset.Text.apply(lambda x: re.sub(r'www\.[a-z]?\.?@[\w]+(com)+|[a-z]+\.(com)', '', x))
        # Lower case all text
        self._pandas_dataset['Text'] = self._pandas_dataset.Text.str.lower()
        # Remove @username
        self._pandas_dataset['Text'] = self._pandas_dataset.Text.apply(lambda x: re.sub('@[^\s]+', '', x))
        # Remove #hashtag
        self._pandas_dataset['Text'] = self._pandas_dataset.Text.apply(lambda x: re.sub('#[A-Za-z0-9_]', '', x))
        # Remove RT (Retweet)
        self._pandas_dataset['Text'] = self._pandas_dataset['Text'].apply(lambda x: re.sub(' RT ', '', x))
        # Fix abbreviations
        self._pandas_dataset['Text'] = self._pandas_dataset['Text'].apply(lambda x: self.cont_to_exp(x))
        # Remove stop words. See https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py to see the full list
        self._pandas_dataset['Text'] = self._pandas_dataset['Text'].apply(
            lambda x: ' '.join([t for t in x.split() if t not in STOP_WORDS]))
        # Remove non-alphabetical characters
        self._pandas_dataset['Text'] = self._pandas_dataset.Text.apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
        # Remove extra space between words
        self._pandas_dataset['Text'] = self._pandas_dataset['Text'].apply(lambda x: ' '.join(x.split()))
        # Removing short words that are 3 characters or less
        self._pandas_dataset['Text'] = self._pandas_dataset['Text'].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    def split_dataset(self):
        self._pandas_dataset = shuffle(self._pandas_dataset).reset_index(drop=True)  # Reset the index after shuffling

        # Splitting the dataset
        self._x = self._pandas_dataset['Text']  # changed from tweet['Tweet'], needed for CNB.fit to work!
        self._y = self._pandas_dataset['Sentiment']
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x, self._y, test_size=0.20,
                                                                                    random_state=19)  # test 20%
        self._tf = TfidfVectorizer(strip_accents='ascii', stop_words='english')
        self._x_train_tf = self._tf.fit_transform(self._x_train)

        # transform the test set with vectoriser
        self._x_test_tf = self._tf.transform(self._x_test)
        self._x_tf = self._tf.transform(self._x)

        with open(Path('../outputs/pickled/cv'), 'wb') as f:
            pickle.dump(self._tf, f)

    # def bert(self):
    #     bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    #     bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    #     import tensorflow_hub as tf
    #     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    #     # preprocessed_text = bert_preprocess(self._tf)
    #     # print(preprocessed_text)
    #     outputs = bert_encoder(self._tf)
    #     l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    #     l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
    #     model = tf.keras.Model(inputs=[text_input], outputs=[l])
    #     model.summary()
    #     METRICS = [
    #         tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    #         tf.keras.metrics.Precision(name='precision'),
    #         tf.keras.metrics.Recall(name='recall')
    #     ]
    #
    #     model.compile(optimizer='adam',
    #                   loss='binary_crossentropy',
    #                   metrics=METRICS)
    #
    #     model.fit(self._x_train_tf, self._y_train, epochs=10)
    #
    #     y_predicted = model.predict(self._x_test_tf)
    #     y_predicted = y_predicted.flatten()
    #
    #     import numpy as np
    #     y_predicted = np.where(y_predicted > 0.5, 1, 0)
    #     print(y_predicted)
    #
    #     sample_dataset = [
    #         'You can win a lot of money, register in the link below',
    #         'You have an iPhone 10, spin the image below to claim your prize and it will be delivered in your door step',
    #         'You have an offer, the company will give you 50% off on every item purchased.',
    #         "Hey Bravin, don't be late for the meeting tomorrow will start lot exactly 10:30 am",
    #         "See you monday, we have alot to talk about the future of this company ."
    #     ]
    #
    #     model.predict(sample_dataset)

    def naive_bayes(self, naive_bayes, naive_bayes_type):
        # Train
        naive_bayes.fit(self._x_train_tf, self._y_train)
        cross_validation = cross_val_score(naive_bayes, self._x_tf, self._y, n_jobs=-1)
        print(naive_bayes_type + ' Cross Validation score = ', cross_validation)
        print(naive_bayes_type + ' Train accuracy = {:.2f}%'.format(
            naive_bayes.score(self._x_train_tf, self._y_train) * 100))
        print(naive_bayes_type + ' Test accuracy = {:.2f}%'.format(
            naive_bayes.score(self._x_test_tf, self._y_test) * 100))
        train = naive_bayes.score(self._x_train_tf, self._y_train)
        test = naive_bayes.score(self._x_test_tf, self._y_test)
        print(naive_bayes_type + ' Confusion Matrix:\n')
        complete = [train, test]
        predict = naive_bayes.predict(self._x_test_tf)  # Predict test
        print(confusion_matrix(self._y_test, predict))  # Print confusion matrix
        print(classification_report(self._y_test, predict))  # Performance check using Complement Naive Bayes

        # PICKLE TRAINED NAIVE_BAYES
        with open(Path('../outputs/Pickled/' + naive_bayes_type + '_model'), 'wb') as f:
            pickle.dump(naive_bayes, f)

        # ACCURACY BAR GRAPH
        label = ['Train Accuracy', 'Test Accuracy']
        plt.xticks(range(len(complete)), label)
        plt.ylabel('Accuracy')
        plt.title(naive_bayes_type + ' Accuracy bar graph for a sample of ' + str(self._dataset_size))
        plt.bar(range(len(complete)), complete, color=['pink', 'black'])
        train_acc = matplotlib_patches.Patch(color='pink', label='Train Accuracy')
        test_acc = matplotlib_patches.Patch(color='black', label='Test Accuracy')
        plt.legend(handles=[train_acc, test_acc], loc='best')
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(Path('../outputs/Accuracy/train_and_test_accuracy_' + naive_bayes_type))
        plt.clf()

        # ROC CURVE
        fpr_dt_1, tpr_dt_1, _ = roc_curve(self._y_test, naive_bayes.predict_proba(self._x_test_tf)[:, 1])
        plt.plot(fpr_dt_1, tpr_dt_1, label='ROC curve ' + naive_bayes_type)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(Path('../outputs/ROC/ROC_curve_' + naive_bayes_type))
        plt.clf()
        roc_score = roc_auc_score(self._y_test, predict)  # Checking performance using ROC Score
        print(naive_bayes_type + ' Area Under the Curve = ' + str(roc_score) + '\n')

        # LIME
        # Converting the vectoriser and model into a pipeline, this is necessary as LIME takes a model pipeline
        # as an input
        pipeline = make_pipeline(self._tf, naive_bayes)
        ls_x_test = list(self._x_test)
        # Saving the class names in a dictionary to increase interpretability
        class_names = {0: 'negative', 1: 'positive'}
        lime_explainer = LimeTextExplainer(class_names=class_names)
        idx = randrange(len(ls_x_test))  # Choose a random single prediction, or use one number here for all to be same
        print(ls_x_test[idx])
        # Explain the chosen prediction, use the probability results of the logistic regression, can also add
        # num_features parameter to reduce the number of features explained
        lime_exp = lime_explainer.explain_instance(ls_x_test[idx], pipeline.predict_proba)

        print('Document id: ' + str(idx))
        print('Tweet: ' + ls_x_test[idx])
        print('Positivity =' + str(pipeline.predict_proba([ls_x_test[idx]]).round(3)[0, 1]))
        print('True class: ' + str(class_names.get(list(self._y_test)[idx])) + '\n')
        lime_exp.save_to_file(Path('../outputs/Lime/lime_' + naive_bayes_type + '.html'))

        lime_exp.as_pyplot_figure()
        plt.savefig(Path('../outputs/Lime/lime_' + naive_bayes_type + '_bargraph'))
        plt.clf()

    def generate_wordcloud(self):
        wordcloud_path = Path('../outputs/wordcloud')
        if not os.path.isdir(wordcloud_path):
            os.makedirs(wordcloud_path)
        text_pos = self._pandas_dataset[self._pandas_dataset['Sentiment'] == 1]  # Only collect text that are positive
        text_neg = self._pandas_dataset[self._pandas_dataset['Sentiment'] == 0]  # Only collect text that are negative

        plt.figure(figsize=(20, 20))
        wc = WordCloud(max_words=2000, width=1600, height=800).generate(' '.join(text_pos['Text']))
        plt.imshow(wc, interpolation='bilinear')
        plt.savefig(Path('../outputs/Wordcloud/word_cloud_positive'))

        plt.figure(figsize=(20, 20))
        wc = WordCloud(max_words=2000, width=1600, height=800).generate(' '.join(text_neg['Text']))
        plt.imshow(wc, interpolation='bilinear')
        plt.savefig(Path('../outputs/Wordcloud/word_cloud_negative'))

    def complement_naive_bayes(self):
        cnb = ComplementNB()
        self.check_dir()
        self.preprocess_dataset()
        self.split_dataset()
        self.naive_bayes(cnb, 'CNB')

    def bernoulli_naive_bayes(self):
        bnb = BernoulliNB()
        self.check_dir()
        self.preprocess_dataset()
        self.split_dataset()
        self.naive_bayes(bnb, 'BNB')

    def multinomial_naive_bayes(self):
        mnb = MultinomialNB()
        self.check_dir()
        self.preprocess_dataset()
        self.split_dataset()
        self.naive_bayes(mnb, 'MNB')

    # def bert(self):
    #     self.check_dir()
    #     self.preprocess_dataset()
    #     self.split_dataset()
    #     self.bert()
