from transformers import pipeline

def load_zero_shot_model(device='cpu'):
    """
    Loads a zero-shot classification model.
    Parameters:
        device (str): The device to load the model on. Default is 'cpu'.
    Returns:
        pipeline: The loaded zero-shot classification model.
    Raises:
        Exception: If there is an error loading the model.
    """

    try:
        return pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def run_zero_shot_model(sequence: str, candidate_labels: list, device='cpu') -> str:

    """
    Runs a zero-shot classification model on the given sequence and candidate labels.
    Args:
        sequence (str): The sequence to classify.
        candidate_labels (list): The list of candidate labels.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
    Returns:
        str: The predicted label.
    Raises:
        Exception: If there is an error running the model.
    """

    model = load_zero_shot_model(device)
    if model is None:
        return "Model could not be loaded."
    try:
        result = model(sequence, candidate_labels)
        return result['labels'][0]
    except Exception as e:
        print(f"Error running model: {e}")
        return "Error running model."