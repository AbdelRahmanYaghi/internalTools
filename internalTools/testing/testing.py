"""
This script is used to test the functions from the modeling module.
"""

import time
# calculate time taken by a function
start=time.time()
from internalTools.modeling import run_ner_model

text = """CR7 plays football !"""
labels = ["person","sport"]

print(run_ner_model(text,labels))

print("Time taken by the scrpit is: ",time.time()-start)