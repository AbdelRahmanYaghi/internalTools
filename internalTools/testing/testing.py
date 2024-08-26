"""
This script is used to test the functions from the modeling module.
"""

import time
# calculate time taken by a function
start=time.time()
# from internalTools.modeling import run_ner_model
from internalTools.modeling import run_qa_model

context = """Cristiano Ronaldo is a Portuguese professional footballer who plays
 as a forward for Premier League club Manchester United and captains the 
 Portugal national team. Often considered the best player in the world and
 widely regarded as one of the greatest players of all time, Ronaldo has won 
 five Ballon d'Or awards and four European Golden Shoes, the most by a 
 European player. He has won 32 trophies in his career, including seven 
 league titles, five UEFA Champions Leagues, one UEFA European Championship 
 and one UEFA Nations League. Ronaldo holds the records for most appearances,
 most goals and assists in the UEFA Champions League, most goals in the UEFA 
 European Championship, most international"""

labels = ["person","sport"]

question = "what is Ronaldos nationality?"
# print(run_ner_model(context,labels))

print(run_qa_model(context,question,device='cuda:0'))
print("Time taken by the scrpit is: ",time.time()-start)