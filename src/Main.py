# Class:        Main.py
# Description:  Driver to run everything
# Author:       R. Liang
# Ver:          11/9/22
from pathlib import Path

from TextClassification import TextClassification
from Bert import Bert
from ImageCaption import ImageCaption

image_caption = ImageCaption()
# image_caption.create_model()
image_caption.run_model(Path('../resources/test-image.jpg'))

# text_classification = TextClassification()
# bert = Bert()
# pred_sentences = ['I am going to shoot up the school', 'I had a lovely time today',
#                   "I'm going to fucking kill everyone here", "I hate everyone here"]
# bert.run_model(pred_sentences)

# text_classification.complement_naive_bayes()
# text_classification.bernoulli_naive_bayes()
# text_classification.multinomial_naive_bayes()
# text_classification.generate_wordcloud()
