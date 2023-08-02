import streamlit as st
import pandas as pd
import re
import time
from tqdm import tqdm
import seaborn as sns
import numpy as np
from textblob import TextBlob # type: ignore
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer # type: ignore
import streamlit as st
import os
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from scipy.spatial.distance import cosine
import torch
from numpy.linalg import norm
#device = torch.device("cuda")

device = torch.device("cpu")
model.to(device)

models_ = r"./IT-Ticket-Issue-Recommender-main/textEmbedding/"

import pickle
# Load the text embeddings from the file
# with open(models_+"text_embeddings.pkl", "rb") as f:
#     embeddings_dataset = pickle.load(f)

# load the X_train from disk

f = models_+"text_embeddings.pkl"
embeddings_dataset = pickle.load(open(f, 'rb'))



# Pre trained Sentence Transformers
from transformers import AutoTokenizer, AutoModel # type: ignore

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

import sys
sys.path.insert(0, r"./IT-Ticket-Issue-Recommender-main/") # retailDynamicPricing\RetailDynamicPricing.py
# sys.path.insert(0, r"./utils/")
#from evaluateModelOnHoldData import evalModel
from ticket_app import itTaskRec


#@st.cache
def main ():
    try:
        st.title("Welcome to IT Ticket Issue Recommender service")

        # if st.checkbox("Retail Churn Test Response", key='1'):
        #     RetailChurnTestResponse()
            
            #RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("IT Ticket Issue Recommender", key='2'):
            itTaskRec(embeddings_dataset, model)
        # if st.checkbox("Retail Churn Dashboard", key='3'):
        #     RetailChurnDashboard()

                
    except Exception as e:
        print(e)
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


