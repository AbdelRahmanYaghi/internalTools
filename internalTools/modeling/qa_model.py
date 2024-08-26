from transformers import pipeline

def load_qa_model(device='cpu'):
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_qa_model(context: str, question: str, device='cpu') -> str:
    model = load_qa_model(device)
    if model is None:
        return "Model could not be loaded."
    try:
        answer = model(question, context)
        return answer['answer']
    except Exception as e:
        print(f"Error running model: {e}")
        return "Error running model." 
