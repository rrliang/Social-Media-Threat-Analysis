# Class:        Bert.py
# Description:  Bert algorithm for text processing
# Author:       R. Liang
# Ver:          11/11/22

from pathlib import Path
import tensorflow_text  # Is needed to run
import tensorflow_hub as hub
import pandas as pd
from keras.saving.save import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

class Bert:

    def __init__(self):
        self._path_to_dataset = Path('../resources/text-post-dataset.csv')

    def create_model(self):
        col_name = ['Sentiment', 'Text']  # Label all the columns
        df = pd.read_csv(self._path_to_dataset, encoding='latin', header=None, names=col_name)

        # print(df['Sentiment'].value_counts())

        df_negative = df[df['Sentiment'] == 0]  # this was df_spam before
        df_positive = df[df['Sentiment'] == 1]  # this was df_ham before

        df_pos_downsampled = df_positive.sample(df_negative.shape[0])
        df_balanced = pd.concat([df_pos_downsampled, df_negative])

        # print(df_balanced['Sentiment'].value_counts())

        X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Sentiment'],
                                                            stratify=df_balanced['Sentiment'])
        map_model_to_preprocess = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'electra_small':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'electra_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_pubmed':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_wiki_books':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }

        map_name_to_handle = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_base/2',
            'electra_small':
                'https://tfhub.dev/google/electra_small/2',
            'electra_base':
                'https://tfhub.dev/google/electra_base/2',
            'experts_pubmed':
                'https://tfhub.dev/google/experts/bert/pubmed/2',
            'experts_wiki_books':
                'https://tfhub.dev/google/experts/bert/wiki_books/2',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
        }

        preprocess = 'bert_en_uncased_L-12_H-768_A-12'
        encoder = 'small_bert/bert_en_uncased_L-8_H-768_A-12'

        bert_preprocess = hub.KerasLayer(map_model_to_preprocess[preprocess])
        bert_encoder = hub.KerasLayer(map_name_to_handle[encoder])

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

        # # PICKLE TRAINED BERT
        # with open(Path('../outputs/Pickled/bert_en_uncased_L-8_H-768_A-12_model'), 'wb') as f:
        #     pickle.dump(model, f)
        model.save(Path('../outputs/Pickled/bert_en_uncased_L-8_H-768_A-12_model.h5'))

        results = model.evaluate(X_test, y_test)

        print(results)

        # y_eval = eval['Text'].values

        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_test, y_predicted))

        print('small_bert/bert_en_uncased_L-8_H-768_A-12' + ' Confusion Matrix:\n')
        cm = confusion_matrix(y_test, y_predicted)
        print(cm)
        print(classification_report(y_test, y_predicted))
        # _tf = TfidfVectorizer(strip_accents='ascii', stop_words='english')
        # x_train_tf = _tf.fit_transform(X_train)
        #
        # # transform the test set with vectoriser
        # x_test_tf = _tf.transform(X_test)
        # x_tf = _tf.transform(df_balanced['Text'])

        # cross_validation = cross_val_score(model, x_tf, df_balanced['Text'], n_jobs=-1)
        # print('small_bert/bert_en_uncased_L-8_H-768_A-12' + ' Cross Validation score = ', cross_validation)
        # print('small_bert/bert_en_uncased_L-8_H-768_A-12' + ' Train accuracy = {:.2f}%'.format(
        #     model.score(x_train_tf, y_train) * 100))
        # print('small_bert/bert_en_uncased_L-8_H-768_A-12' + ' Test accuracy = {:.2f}%'.format(
        #     model.score(x_test_tf, y_test) * 100))
        # train = model.score(x_train_tf, y_train)
        # test = model.score(x_test_tf, y_test)
        # print('small_bert/bert_en_uncased_L-8_H-768_A-12' + ' Confusion Matrix:\n')
        # complete = [train, test]
        # predict = model.predict(self.x_test_tf)  # Predict test
        # print(confusion_matrix(self._y_test, predict))  # Print confusion matrix
        # print(classification_report(self._y_test, predict))  # Performance check using Complement Naive Bayes



        # # ACCURACY BAR GRAPH
        # label = ['Train Accuracy', 'Test Accuracy']
        # plt.xticks(range(len(complete)), label)
        # plt.ylabel('Accuracy')
        # plt.title(naive_bayes_type + ' Accuracy bar graph for a sample of ' + str(self._dataset_size))
        # plt.bar(range(len(complete)), complete, color=['pink', 'black'])
        # train_acc = matplotlib_patches.Patch(color='pink', label='Train Accuracy')
        # test_acc = matplotlib_patches.Patch(color='black', label='Test Accuracy')
        # plt.legend(handles=[train_acc, test_acc], loc='best')
        # plt.gcf().set_size_inches(10, 10)
        # plt.savefig(Path('../outputs/Accuracy/train_and_test_accuracy_' + naive_bayes_type))
        # plt.clf()
        #
        # # ROC CURVE
        # model
        # fpr_dt_1, tpr_dt_1, _ = roc_curve(self._y_test, model.predict_proba(self._x_test_tf)[:, 1])
        # plt.plot(fpr_dt_1, tpr_dt_1, label='ROC curve ' + naive_bayes_type)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend()
        # plt.gcf().set_size_inches(8, 8)
        # plt.savefig(Path('../outputs/ROC/ROC_curve_' + naive_bayes_type))
        # plt.clf()
        # roc_score = roc_auc_score(self._y_test, predict)  # Checking performance using ROC Score
        # print(naive_bayes_type + ' Area Under the Curve = ' + str(roc_score) + '\n')
        #
        # # LIME
        # # Converting the vectoriser and model into a pipeline, this is necessary as LIME takes a model pipeline
        # # as an input
        # pipeline = make_pipeline(self._tf, naive_bayes)
        # ls_x_test = list(self._x_test)
        # # Saving the class names in a dictionary to increase interpretability
        # class_names = {0: 'negative', 1: 'positive'}
        # lime_explainer = LimeTextExplainer(class_names=class_names)
        # idx = randrange(len(ls_x_test))  # Choose a random single prediction, or use one number here for all to be same
        # print(ls_x_test[idx])
        # # Explain the chosen prediction, use the probability results of the logistic regression, can also add
        # # num_features parameter to reduce the number of features explained
        # lime_exp = lime_explainer.explain_instance(ls_x_test[idx], pipeline.predict_proba)
        #
        # print('Document id: ' + str(idx))
        # print('Tweet: ' + ls_x_test[idx])
        # print('Positivity =' + str(pipeline.predict_proba([ls_x_test[idx]]).round(3)[0, 1]))
        # print('True class: ' + str(class_names.get(list(self._y_test)[idx])) + '\n')
        # lime_exp.save_to_file(Path('../outputs/Lime/lime_' + naive_bayes_type + '.html'))
        #
        # lime_exp.as_pyplot_figure()
        # plt.savefig(Path('../outputs/Lime/lime_' + naive_bayes_type + '_bargraph'))
        # plt.clf()

    def run_model(self, pred_sentences):
        model = load_model(Path('../outputs/Pickled/bert_en_uncased_L-8_H-768_A-12_model.h5'),
                           custom_objects={'KerasLayer': hub.KerasLayer})
        predicted = model.predict(pred_sentences)
        for i in range(len(pred_sentences)):
            print(pred_sentences[i])
            num = predicted[i]
            if num > .5:
                print("Positive: " + str(predicted[i]))
            else:
                print("Negative: " + str(predicted[i]))