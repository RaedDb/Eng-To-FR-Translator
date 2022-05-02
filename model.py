import os

# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model

from urllib.request import urlretrieve

import logging

import pickle
import json

class TranslatorClassifier:

    def __init__(self, model_path):
        logging.info("TranslatorClassifier class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")
        with open('eng_tokenizer.pickle', 'rb') as handle:
            self.eng_tokenizer = pickle.load(handle)
        with open('fr_tokenizer.pickle', 'rb') as handle:
            self.fr_tokenizer = pickle.load(handle)

        with open('sentences_len.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        self.eng_len = json_object['eng_len']
        self.fr_len = json_object['fr_len']


    def predict(self, sentence):
        from keras.preprocessing.sequence import pad_sequences
        import numpy as np

        y_id_to_word = {value: key for key, value in self.fr_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'

        sentence = self.eng_tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=self.eng_len, padding='post')
        # predict the sentence which is in english
        predictions = self.model.predict(sentence)

        result = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])

        #returns the predicted sentece which is in french
        return result

def main():
    model = TranslatorClassifier('machine_translation_model.h5')

    predicted_class = model.predict("Hello there!")
    logging.info("The french translated sentence with respect to our model is:\n {}".format(predicted_class))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()