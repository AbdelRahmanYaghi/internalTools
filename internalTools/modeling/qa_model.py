from transformers import pipeline

def load_qa_model(device='cpu'):
    """
    Loads a question-answering model.
    Parameters:
        device (str): The device to load the model on. Default is 'cpu'.
    Returns:
        pipeline: The loaded question-answering model.
    Raises:
        Exception: If there is an error loading the model.
    """

    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_qa_model(context: str, question: str, device='cpu') -> str:
    """
    Runs a question answering model on the given context and question.
    Args:
        context (str): The context or passage to search for the answer.
        question (str): The question to be answered.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
    Returns:
        str: The answer to the question.
    Raises:
        Exception: If there is an error running the model.
    """

    model = load_qa_model(device)
    if model is None:
        return "Model could not be loaded."
    try:
        answer = model(question, context)
        return answer['answer']
    except Exception as e:
        print(f"Error running model: {e}")
        return "Error running model." 
