import torch
import torchvision
from transformers import pipeline
from datetime import datetime

print(torch.__version__)
print(torchvision.__version__)

with open('./association_model/association_dict.txt', 'r') as dict_file:
    CANDIDATE_LABELS = dict_file.read().split("\n")

# CANDIDATE_LABELS = ['calm', 'people', 'plant']

MODEL_NAME = 'cross-encoder/nli-distilroberta-base'

MODEL = pipeline("zero-shot-classification", model=MODEL_NAME)


class ModelLogic:
    def __init__(self):
        pass

    def model_specific_logic(self, sequence_to_classify):
        started_time = datetime.now()
        print(sequence_to_classify)
        print(CANDIDATE_LABELS)
        predict = MODEL(sequence_to_classify, CANDIDATE_LABELS, multi_label=True)
        print(predict)

        result = [predict['labels'][i] for i in range(len(predict['labels'])) if predict['scores'][i] > 0.9]
        print(result)

        ended_time = datetime.now()
        print(f'TIME:{ended_time - started_time}')

        return [{'name': tag} for tag in result]
