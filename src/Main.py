# Class:        Main.py
# Description:  Driver to run everything
# Author:       R. Liang
# Ver:          11/9/22

from pathlib import Path

from TextClassification import TextClassification
from Bert import Bert
from ImageCaption import ImageCaption
import pandas as pd
import numpy as np
import os
import winsound

image_caption = ImageCaption()
bert = Bert()
# Only for naive bayes classifiers
# text_classification = TextClassification()

# No need to create bert model if h5 file has already been generated
# bert.create_model()

# No need to create image model if h5 file has already been generated
# image_caption.create_model()

########################################################################################################################
# Commented lines of code below this creates new pickeled models from each naive bayes classifier
########################################################################################################################
# text_classification.complement_naive_bayes()
# text_classification.bernoulli_naive_bayes()
# text_classification.multinomial_naive_bayes()
# text_classification.generate_wordcloud()


########################################################################################################################
# These uncommented lines of code takes a folder of test pictures, as well as an xlsx file with said pictures (with
# extension name + the textual component of the social media post linked with that image (see example file in ../testing
# images dir; essentially running the multimodal model on the entire list of images and texts.
########################################################################################################################
# path = r'..\testing images'
# xl = r"..\testing images\testing - Sheet1.xlsx"
#
# xl_frame = pd.read_excel(xl)
# print(type(xl_frame))
# arr = xl_frame.to_numpy()
# print(len(arr))
#
# IMAGE = 0
# TEXT = 1
# ACTUAL = 2
# PREDICTED = 3
# PRED_NUM = 4
# GENERATED_TEXT = 5
# blah = 0
# errorspot = 0
#
# FINAL_TEST_CSV_FILE_NAME = "finished6.csv"
# if os.path.isfile(FINAL_TEST_CSV_FILE_NAME):
#     os.remove(FINAL_TEST_CSV_FILE_NAME)
#
# try:
#     for i in range(len(arr)):
#         errorspot = 1
#         image_path = path + "\\" + arr[i][IMAGE]
#         errorspot = 2
#         image = image_caption.run_model(Path(image_path))
#         errorspot = 3
#         text = arr[i][TEXT]
#         errorspot = 4
#         post = text + ' ' + image
#         errorspot = 5
#         pred_sentences = [post]
#         errorspot = 6
#         predicted_num = bert.run_model(pred_sentences)
#         errorspot = 7
#         if predicted_num > .5:
#             errorspot = 8
#             predicted = "Positive"
#             errorspot = 9
#         else:
#             predicted = "Negative"
#             errorspot = 10
#         # arr[i][TEXT] = '""' + str(text) + '""'
#         errorspot = 11
#         arr[i][PREDICTED] = predicted
#         errorspot = 12
#         arr[i][PRED_NUM] = predicted_num
#         errorspot = 13
#         arr[i][GENERATED_TEXT] = image
#         errorspot = 14
#         blah = i
#         errorspot = 15
#
#         arr[i][TEXT] = None
#
#         with open(FINAL_TEST_CSV_FILE_NAME, "a", encoding="utf-8") as f:
#             errorspot = 16
#             np.savetxt(f, [arr[i]], fmt="%s", delimiter = ',', newline='\n')
#             errorspot = 17
#
# except Exception as e:
#     print("Row " + str(i) + " did not work due to " + str(e) + ". Error spot: " + str(errorspot))
#     winsound.Beep(500, 1000)


## Below is just to run image captioning
image = image_caption.run_model(Path('augmentedschoolshooter66.jpg'))
print("Image caption: " + image)
pred_sentences = [image]

########################################################################################################################
# These commented lines of code takes a manual path of an image and a string of the social media caption, and returns
# the result of the multimodal model
########################################################################################################################

# text = "Mom & son day testing out his new Xmas present. My first time shooting a 9mm I hit the bullseye."
# image = image_caption.run_model(Path('../resources/post.png'))
# print("Textual component: " + text)
# print("Image caption: " + image)
# post = text + ' ' + image
# pred_sentences = [post]
#
# bert.run_model(pred_sentences)
# print(text_classification.predictText(post))

########################################################################################################################
# This commented line of code just tests the bert model on each manually entered text in a list
########################################################################################################################
# bert.run_model(["I am going to shoot up the school", "I had a lovely time today", "I'm going to kill everyone here"])



