from collections import defaultdict
from gliner import GLiNER

ner_model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5")

def run_ner_model(text:str,labels:list)->dict:
    """
    Run the NER model on the given text.

    Usage:
    '''
    run_ner_model("CR7 plays football !",["person","sport"])
    '''

    Output:

    {
        "person": ["CR7"],
        "sport": ["football"]
    }

    """
    
    entities = ner_model.predict_entities(text, labels)


    ner_dict = defaultdict(set)

    for entity in entities:
        ner_dict[entity["label"]].add(entity["text"])

    ner_dict = {label: list(texts) for label, texts in ner_dict.items()}

    return ner_dict
    