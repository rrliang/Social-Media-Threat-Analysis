from pathlib import Path
import tensorflow_text #Is needed to run
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class Bert:
    def __init__(self):
        self._path_to_dataset = Path('../resources/text-post-dataset.csv')
        col_name = ['Sentiment', 'Text']  # Label all the columns
        df = pd.read_csv(self._path_to_dataset, encoding='latin', header=None, names=col_name)

        print(df['Sentiment'].value_counts())

        df_negative = df[df['Sentiment'] == 0]  # this was df_spam before
        df_positive = df[df['Sentiment'] == 1]  # this was df_ham before

        df_pos_downsampled = df_positive.sample(df_negative.shape[0])
        df_balanced = pd.concat([df_pos_downsampled, df_negative])

        print(df_balanced['Sentiment'].value_counts())

        X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Sentiment'],
                                                            stratify=df_balanced['Sentiment'])

        import tensorflow as tf

        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
        l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
        model = tf.keras.Model(inputs=[text_input], outputs=[l])
        model.summary()
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=METRICS)

        model.fit(X_train, y_train, epochs=10)

        y_predicted = model.predict(X_test)
        y_predicted = y_predicted.flatten()
        y_predicted = np.where(y_predicted > 0.5, 1, 0)

        sample_dataset = [
            'You can win a lot of money, register in the link below',
            'You have an iPhone 10, spin the image below to claim your prize and it will be delivered in your door step',
            'You have an offer, the company will give you 50% off on every item purchased.',
            "Hey Bravin, don't be late for the meeting tomorrow will start lot exactly 10:30 am",
            "See you monday, we have alot to talk about the future of this company ."
        ]

        print(model.predict(sample_dataset))
