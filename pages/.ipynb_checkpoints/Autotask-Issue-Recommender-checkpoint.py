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

models_ = r"./AutotaskIssueRecommender-main/textEmbedding/"

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
sys.path.insert(0, r"./AutotaskIssueRecommender-main/") # retailDynamicPricing\RetailDynamicPricing.py
# sys.path.insert(0, r"./utils/")
#from evaluateModelOnHoldData import evalModel
from autotask_app import autoTaskRec


@st.cache_resource
def main ():
    try:
        st.title("Welcome to Autotask Issue Recommender service")

        # if st.checkbox("Retail Churn Test Response", key='1'):
        #     RetailChurnTestResponse()
            
            #RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("Autotask Issue Recommender", key='2'):
            autoTaskRec(embeddings_dataset, model)
        # if st.checkbox("Retail Churn Dashboard", key='3'):
        #     RetailChurnDashboard()

                
    except Exception as e:
        print(e)
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


