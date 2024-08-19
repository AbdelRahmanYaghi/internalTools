'''
Functions related to modeling.
'''
import os
from sentence_transformers import SentenceTransformer

def load_model(model_name, model_path = 'downloaded_models', **kwargs):
    """
    Down/Load a model using the SentenceTranformer Model.

    Usage:
    '''
    model = load_model("mixedbread-ai/mxbai-embed-large-v1")
    model.encode("Hello there!")
    '''
    """

    if model_path != 'downloaded_models':
        if os.path.exists(os.path.join(model_name.split('/')[-1], 'config.json')):
            model = SentenceTransformer(model_path, **kwargs)
        else:
            model = SentenceTransformer(model_name, **kwargs)
            model.save(model_path)
    else:
        def_path = os.path.join('downloaded_models', model_name.split('/')[-1])
        if os.path.exists(os.path.join(def_path, 'config.json')):
            model = SentenceTransformer(os.path.join(def_path), **kwargs)
        else:
            model = SentenceTransformer(model_name, **kwargs)
            model.save(os.path.join(def_path))

    return model
