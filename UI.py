from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import torch
import pandas as pd
import numpy as np
import streamlit as st
import re

pd.set_option("display.precision", 2)
pd.set_option("display.max_rows", 40)
st.set_page_config(page_title="Recomenddit", page_icon=None, layout='wide', initial_sidebar_state='auto')

def make_clickable(sub):
    link = "https://www.reddit.com/r/"+sub
    return f'<a target="_blank" href="{link}">{sub}</a>'

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('df_small.csv')
    sub_embs = torch.load('emb_small').float()
    sub_list = df['subreddit'].tolist()
    return df, sub_embs, sub_list

df, sub_embs, sub_list = load_data()

@st.cache(allow_output_mutation=True)
def load_all():
    model = AutoModel.from_pretrained('deepset/sentence_bert')
    tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
    return model, tokenizer

model, tokenizer = load_all()


st.title('Subreddit Search Engine')

nsfw_df = pd.DataFrame({
  'nsfw': ['No', 'Yes', 'Both']
})

valmin = st.sidebar.number_input('Minimum Subscribers', min_value = 5000, value = 50000, step = 10000)
valmax = st.sidebar.number_input('Maximum Subscribers', min_value = valmin, value = 50000000, step = 10000)

option = st.sidebar.selectbox(
    'Show NSFW Subreddits',
     nsfw_df['nsfw'])

disp_count = st.sidebar.slider('Number of Recommendations', 10, 100, (50), 1)
accuracy = st.sidebar.slider('Accuracy Slider', 10, 100, (50), 1, help='The degree to which subreddits are included that have very little collected data, 100 is maximally accurate')

st.sidebar.markdown("Made by [Magnus Petersen](https://magnuspetersen.github.io/portfolio/)<br/><br/> The method works by comparing the vector resulting from embedding text, using [S-BERT](https://arxiv.org/abs/1908.10084), for each subreddit and comparing it with the query text embedding using cosine similarity.<br/><br/> The data was gathered with the Reddit API by collecting the titles and text of the top post of each subreddit.", unsafe_allow_html=True)

sentence = st.text_input('Search Input: Works best if you input something you would expect to find on the Subreddit you are looking for, like a title.', 'Here is my favorite Chinese baking recipe')


# run inputs through model and mean-pool over the sequence
# dimension to get sequence-level representations
inputs = tokenizer.batch_encode_plus([sentence],
                                     return_tensors='pt',
                                     pad_to_max_length=True)

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
output = model(input_ids, attention_mask=attention_mask)[0]
output = output.mean(dim=1)
# now find the labels with the highest cosine similarities to
# the sentence
similarities = F.cosine_similarity(output, sub_embs)
closest = similarities.argsort(descending=True)

res_df = df.iloc[closest]
closest = closest.detach().numpy()
res_df['similarity'] = similarities[closest].detach()
res_df = res_df[res_df['subscribers'] > int(valmin)]
res_df = res_df[res_df['subscribers'] < int(valmax)]
res_df = res_df[res_df['textlength'] > accuracy/100*3000]

if option == 'No':
    res_df = res_df[res_df['nsfw'] == 0]
if option == 'Yes':
    res_df = res_df[res_df['nsfw'] == 1]

res_df['subscribers'] = res_df['subscribers'].astype(int)
res_df['similarity']=(res_df['similarity']-res_df['similarity'].min())/(res_df['similarity'].max()-res_df['similarity'].min())
res_df['subreddit'] = res_df['subreddit'].apply(make_clickable)
res_df = res_df.iloc[0:disp_count].reset_index(drop=True)
res_df = res_df[['subreddit', 'subscribers', 'similarity']]
res_df = res_df.rename(columns={'subreddit': 'Subreddit', 'subscribers': 'Subscribers', 'similarity': 'Similarity'})
res_df = res_df.to_html(escape=False)
res_df = re.sub('(<th>Subreddit)', r'\1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;', res_df)
st.write(res_df, unsafe_allow_html=True)
