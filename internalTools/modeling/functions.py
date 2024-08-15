from sentence_transformers import SentenceTransformer
import os

def load_model(model_name, model_path = 'downloaded_models'):
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
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer(model_name)
            model.save(model_path)
    else:
        if os.path.exists(os.path.join('downloaded_models', model_name.split('/')[-1], 'config.json')):
            model = SentenceTransformer(os.path.join('downloaded_models', model_name.split('/')[-1]))
        else:
            model = SentenceTransformer(model_name)
            model.save(os.path.join('downloaded_models', model_name.split('/')[-1]))

    return model

