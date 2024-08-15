import string
from .constants import emoji_translation_table
from nltk import word_tokenize
import ftfy
    
def process_text(text, remove_emojis = False, remove_punctuation = False, stopwords: list = None, stemmer = None, tokenizer = word_tokenize, fix_encoding = True):

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