from operator import mod
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local_css("style.css")
import base64

st.set_page_config(
    page_title="AIJobThreat",
    page_icon=":clipboard:",
    layout="wide",
    initial_sidebar_state="auto"
)

#code assist to embed local image as background: https://stackoverflow.com/questions/72582550/how-do-i-add-background-image-in-streamlit
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

model = joblib.load(open('../pickles/Model.pkl','rb'))
km_tools = joblib.load(open('../pickles/Km_tools.pkl','rb'))
tools = joblib.load(open('../pickles/tools.pkl','rb'))
skills = joblib.load(open('../pickles/skills.pkl','rb'))
X = joblib.load(open('../pickles/Xfull.pkl','rb'))

st.title("Protect your career from AI threat")

st.subheader('What jobs can help you lower AI threat to your career?')

st.markdown(
    "Please fill in the fields below, to learn: 1/ how your job is impacted by AI. 2/ similar jobs you can consider, in case you want to lower risk from AI. Please be sure you separate entries by commas."
)

text_jobs = st.text_area('Identify the job(s) you are interested in pursuing: ').strip()
text_tasks = st.text_area('Identify the task(s) you are energized to perform as part of your job: ').strip()
text_tools = st.text_area('Identify the tool(s) you have experience using: ').strip()
text_skills = st.selectbox('Select the skill(s) you are excited to leverage: ', skills.columns, index=0)


#def text_process(text):
#  stops = nltk.corpus.stopwords.words('english')
#  new_stop_words = ["whom"]
#  stops.extend(new_stop_words)
  
#  w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
#  lemmatizer = WordNetLemmatizer()

#  words = [x.lower() for x in text.split() if (x not in stops) and (len(x)>1)]
#  sentence = " ".join(words)
#  splits = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(sentence)]
#  list_of_sentences = " ".join(splits)

#  return list_of_sentences

#text_job = text_process(text_jobs)
#text_task = text_process(text_tasks)
#text_tool = text_process(text_tools)

Xstreamlit = pd.DataFrame(columns = X.columns, index=[0])
Xstreamlit.fillna(0, inplace=True)

for i in range(len(text_skills)):
  if text_skills[i] in Xstreamlit.columns:
    Xstreamlit.loc[0][text_skills[i]] = 1

combinedtext_ = [text_process(text_tasks),text_process(text_job), text_process(text_tools)]
combinedtext = "|".join([str(item) for item in combinedtext_])

Xstreamlit["combinedtext"] = combinedtext

prediction = model.predict(Xstreamlit)[0]
if prediction == 0:
  probability = model.predict_proba(Xstreamlit)[0][0]
  print(f"AI poses a HIGH risk to your selected job.")
elif prediction == 1:
  probability = model.predict_proba(Xstreamlit)[0][1]
  print(f"AI poses a MODERATE risk to your selected job.")
else:
  probability = model.predict_proba(Xstreamlit)[0][2]
  print(f"AI poses a LOW risk to your selected job.")

#Clusters
#Tools
Xstreamlit_tools = pd.DataFrame(columns = tools.columns, index=["input", "match", "fuzzy_match"])
Xstreamlit_tools.fillna(0, inplace=True)
tools_list = list(text_tools.split(", "))

for i in range(len(tools_list)):
    j = tools_list[i]
    if len(difflib.get_close_matches(j,Xstreamlit_tools.columns)) != 0:
        k = difflib.get_close_matches(j,Xstreamlit_tools.columns, cutoff=0.8)
        Xstreamlit_tools.loc["input",k] = 1
        Xstreamlit_tools.loc["match",k] = tools_list[i]
        Xstreamlit_tools.loc["fuzzy_match",k] = k[0]

cluster = km_tools.predict(Xstreamlit_tools.head(1))[0]
tools["cluster"]= km_tools.labels_
top_tools_jobs = pd.DataFrame(tools[tools["cluster"]==cluster].head(3).index)
