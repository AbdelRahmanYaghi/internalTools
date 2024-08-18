'''
Functions related to text processing.
'''

import string
import ftfy
from nltk import word_tokenize
from .constants import emoji_translation_table

def process_text(text,
                remove_emojis = False,
                remove_punctuation = False,
                stopwords: list = None,
                stemmer = None,
                tokenizer = word_tokenize,
                fix_encoding = True):
    '''
    text (str): Text to process.
    remove_emojis (bool): = False. Removes emojis form the text.
    remove_punctuation (bool): = False. Remomves punctuation for the text.
    stopwords (list[str]): = None. Removes the given list of stopwords for the text.
    stemmer (function.call): = None. Applies stemming.
    tokenizer (function.call): = nltk.word_tokenize. Applies tokenization to the text.
    fix_encoding (bool): = True. Applies ftfy.fix_encoding.
    '''

    if fix_encoding:
        text = ftfy.fix_encoding(text)

    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    if remove_emojis:
        text = text.translate(str.maketrans(emoji_translation_table))

    text = tokenizer(text)

    if stopwords:
        text = ' '.join([word for word in text if word not in stopwords])

    if stemmer:
        if isinstance(text, list):
            text = list(map(stemmer, text))
        else:
            text = stemmer(text)

    return text
