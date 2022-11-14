# Class:        Main.py
# Description:  Driver to run everything
# Author:       R. Liang
# Ver:          11/9/22

from TextClassification import TextClassification
from Bert import Bert

text_classification = TextClassification()
bert = Bert()
# text_classification.complement_naive_bayes()
# text_classification.bernoulli_naive_bayes()
# text_classification.multinomial_naive_bayes()
# text_classification.generate_wordcloud()
