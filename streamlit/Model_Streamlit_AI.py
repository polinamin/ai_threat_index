from operator import mod
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import difflib
#from st_aggrid import AgGrid, GridOptionsBuilder
import base64
from sklearn.preprocessing import LabelEncoder, StandardScaler
#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="AIJobThreat",
    layout="wide",
    initial_sidebar_state="auto"
)


st.markdown('''
<style>
div[class*="stTextArea"] label {
  font-size: 32px;
  color: white;
  text-transform: uppercase;
  background: #045D5F;
  font-family: 'Roboto', sans-serif;
  font-weight: 1000;
  line-height: 3.0;
  text-indent: 1em;

div[class*="stMulitSelect"] label {
  font-size: 32px;
  color: white;
  text-transform: uppercase;
  font-weight: bold;
  line-height: 3.0;
  text-indent: 1em;
  height: fit-content;
  white-space: normal; 
  max-width: 100%; 
  overflow-wrap: anywhere;

div[class*="stError"] label {
  font-size: 20px;
  background:#ffa080;

div[class*="stWarning"] label {
  font-size: 20px;
  background:	#ffff80;

div[class*="stSuccess"] label {
  font-size: 20px;
  background:	#a5d46a;
            
</style>
''',unsafe_allow_html=True)


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpeg_file):
    bin_str = get_base64(jpeg_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-size: cover;
    background-repeat:no-repeat;
    #margin-top: 200px;
    }
    </style> 
    '''% bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_background("../Images/blurry_background.jpeg")


model = joblib.load(open('../pickles/Model.pkl','rb'))
km_tools = joblib.load(open('../pickles/Km_tools.pkl','rb'))
tools = joblib.load(open('../pickles/tools_.pkl','rb'))
skills_ = joblib.load(open('../pickles/skills_f.pkl','rb'))
skills = skills_.drop(columns = ["rank_new", "Title", "cluster","count"])
industrycols = joblib.load(open('../pickles/industry_cols.pkl','rb'))
km_industry = joblib.load(open('../pickles/km_industries.pkl','rb'))
industry_ = joblib.load(open('../pickles/industry_.pkl','rb'))
industry = industry_.drop(columns = ["rank_new", "Title", 'Projected Growth (2022-2032)', 'Projected Job Openings (2022-2032)', "cluster"])
km_skills = joblib.load(open('../pickles/Km_skills.pkl','rb'))
X = joblib.load(open('../pickles/Xfull.pkl','rb'))
imputer = joblib.load(open('../pickles/imputer.pkl','rb'))

st.header('Protect your career from AI threat', divider='rainbow')


col1, col2 = st.columns(2)
with col1:
  st.subheader('What jobs can help you lower AI threat to your career?')
  st.subheader("Please fill in the fields below, to learn how your job and career preferences are impacted by AI. Please separate each entry by a comma.")
with col2:
  col3, col4, col5 = st.columns(3)
  with col3:
    st.write("")
  with col4:
    st.image('../Images/background_image.jpeg', width=400)
  with col5:
    st.write("")
    

text_jobs = st.text_area('Identify the job(s) you are interested in pursuing: ').strip()
text_tasks = st.text_area('Identify the task(s) you are energized to perform as part of your job: ').strip()
text_tools = st.text_area('Identify the tool(s) you have experience using: ').strip()
text_skills = st.multiselect('Select the skill(s) you are excited to leverage: ', skills.columns, 
                           placeholder="Select your skills...", max_selections=5)
#st.write(text_skills)
text_industry = st.multiselect('Indicate the industry or industries where you would like to work: ', industrycols, 
                           placeholder="Select your industry...", max_selections=3)


#text_job = text_process(text_jobs)
#text_task = text_process(text_tasks)
#text_tool = text_process(text_tools)

Xstreamlit_transformed = pd.DataFrame(columns = X.columns, index=[0])
Xstreamlit_transformed.fillna(0.0, inplace=True)

Xstreamlit = Xstreamlit_transformed.drop(columns="combinedtext")

for i in range(len(text_skills)):
  if text_skills[i] in Xstreamlit.columns:
    Xstreamlit.loc[0][text_skills[i]] = X[text_skills[i]].mean()

ss = StandardScaler()
Xstreamlit_s = ss.fit_transform(Xstreamlit)
Xstreamlit[:] = imputer.transform(Xstreamlit_s)

combinedtext_ = [(text_tasks), (text_jobs), (text_tools), (text_industry)] 
combinedtext = "|".join([str(item) for item in combinedtext_])

Xstreamlit.loc[:,"combinedtext"] = combinedtext
prediction = model.predict(Xstreamlit)


if st.button('Submit'):
  if len(text_jobs)==len(text_tasks) and len(text_jobs)==len(text_tools) and len(text_jobs) == len(text_skills) and len(text_jobs) == len(text_industry):
   st.write("Please enter your criteria above.")
  elif prediction == 1:
    st.success(f"AI poses a **LOW** risk to your selected job.")
  elif prediction == 2:
    st.warning(f"AI poses a **MODERATE** risk to your selected job.")
  elif prediction == 3:
    st.error(f"AI poses a **HIGH** risk to your selected job.")
  else:
    st.error(f"AI poses a **VERY HIGH** risk to your selected job.")


dict = {1:"Low Impact", 2: "Moderate Impact", 3: "High Impact", 4: "Very High Impact"}

#Clusters
#Tools
tools_ = tools.drop(columns=["Title", "rank_new", "cluster"])
Xstreamlit_tools = pd.DataFrame(columns = tools_.columns, index=["input", "match", "fuzzy_match"])
Xstreamlit_tools.fillna(0, inplace=True)
tools_list = text_tools.split(',')
tools_list_ = []

for i in range(len(tools_list)):
  j = tools_list[i].strip()
  if len(difflib.get_close_matches(j,Xstreamlit_tools.columns, n=1, cutoff=0.7)) !=0:
    k = difflib.get_close_matches(j,Xstreamlit_tools.columns, n=1, cutoff=0.7)[0]
    Xstreamlit_tools.loc["input",k] = 1
    Xstreamlit_tools.loc["match",k] = j
    Xstreamlit_tools.loc["fuzzy_match",k] = k
    tools_list_.append(k)
    if k is None:
        st.write("Unfortunately, there are no job matches based on your preferred tools in our system. Please enter a different tool.")


cluster_tool = km_tools.predict(Xstreamlit_tools.head(1))[0]
tools["cluster"]= km_tools.labels_
top_tools_jobs = pd.DataFrame(tools[tools["cluster"]==cluster_tool].head(3).index).values

tools["AI Impact"] = tools["rank_new"].map(dict)

if tools_list == []:
  st.write("")
elif text_tools == "":
  st.write("")
elif tools_list_ != []:
  tools_selected = tools[tools_list_]
  cols_sum = tools_selected.iloc[:,0]
  for i in range(1, len(tools_selected.columns)):
      cols_sum = cols_sum + tools_selected.iloc[:,i]
  tools_selected["select_col"] = cols_sum == 0
  tools["select_col"] = tools_selected["select_col"]
  st.write(f"The jobs with the lowest AI Impact, based on the tools you enjoy using are:")
  st.toolsdf=st.dataframe(
   (tools[(tools["cluster"]==cluster_tool) & (tools["select_col"] == False)][["Title", "rank_new","AI Impact"]].sort_values(by="rank_new").head(3).set_index("Title")).style.set_properties(**{"background-color": "white", "color": "black"}).format({"rank_new": "{:.2f}"})
) 
  if tools_selected.empty:
    tools_selected["select_col"]="NA"
    tools["select_col"]= tools_selected["select_col"]
    st.write("Unfortunately, there are no job matches based on your preferred tools in our system. Please enter a different tool.")

  tools.drop(columns=["select_col"])

#Clusters
#Industries
Xstreamlit_industries = pd.DataFrame(columns = industry.columns, index=["input", "match", "fuzzy_match"])
Xstreamlit_industries.fillna(0, inplace=True)
industry_list = text_industry
industry_list_ = []

for i in range(len(industry_list)):
    j = industry_list[i]

    if len(difflib.get_close_matches(j,Xstreamlit_industries.columns, cutoff=0.7)) != 0:
      k = difflib.get_close_matches(j,Xstreamlit_industries.columns, cutoff=0.7)[0]
      Xstreamlit_industries.loc["input",k] = 1
      Xstreamlit_industries.loc["match",k] = j
      Xstreamlit_industries.loc["fuzzy_match",k] = k
      industry_list_.append(k)
    if k is None:
      st.write("Unfortunately, there are no job matches based on your selected industries in our system. Please enter a different industry.")
  

cluster_industry = km_industry.predict(Xstreamlit_industries.head(1))[0]
industry_["cluster"]= km_industry.labels_
top_industries_jobs = pd.DataFrame(industry_[industry_["cluster"]==cluster_industry].head(3).index).values

industry_["AI Impact"] = industry_["rank_new"].map(dict)
industry_selected = industry[industry_list_]


if text_industry == "":
  st.write("")
elif industry_list_ == []:
  st.write("")
else:
  cols_sum = industry_selected.iloc[:,0]
  for i in range(1, len(industry_selected.columns)):
      cols_sum = cols_sum + industry_selected.iloc[:,i]
  industry_selected["select_col"] = cols_sum == 0
  industry_["select_col"] = industry_selected["select_col"]
  st.write(f"The jobs with the lowest AI Impact, based on the industry/industries you selected:")
  st.industrydf=st.dataframe(
  (industry_[(industry_["cluster"]==cluster_industry) & (industry_["select_col"] == False)][["Title", "rank_new","AI Impact",
                                                                                             'Projected Growth (2022-2032)', 'Projected Job Openings (2022-2032)']].sort_values(by="rank_new").head(3).set_index("Title")).style.set_properties(**{"background-color": "white","color": "black"}).format({"rank_new": "{:.2f}", "Projected Job Openings (2022-2032)": "{:.0f}"})
)
  industry_.drop(columns=["select_col"])


#Clusters
#Skills
Xstreamlit_skills = pd.DataFrame(columns = skills.columns, index=["input", "match", "fuzzy_match"])
Xstreamlit_skills.fillna(0, inplace=True)
skills_list = text_skills
skills_list_ = []

for i in range(len(skills_list)):
    j = skills_list[i]
    if len(difflib.get_close_matches(j,Xstreamlit_skills.columns, cutoff=0.7))!= 0:
        k = difflib.get_close_matches(j,Xstreamlit_skills.columns, cutoff=0.7)[0]
        Xstreamlit_skills.loc["input",k] = skills[j].mean()
        Xstreamlit_skills.loc["match",k] = j
        Xstreamlit_skills.loc["fuzzy_match",k] = k[0]
        skills_list_.append(k)
    if k is None:
      st.write("Unfortunately, there are no job matches based on your selected skills in our system. Please update your selection.")


cluster_skills = km_skills.predict(Xstreamlit_skills.head(1))[0]
skills_["cluster"]= km_skills.labels_
top_skills_jobs = pd.DataFrame(skills_[skills_["cluster"]==cluster_skills]["Title"]).head(3)

skills_["AI Impact"] = skills_["rank_new"].map(dict)
skills_selected = skills[skills_list_]

if text_skills == "":
  st.write("")
elif skills_list_ == []:
  st.write("")
else:
  cols_sum = skills_selected.iloc[:,0]
  for i in range(1, len(skills_selected.columns)):
      cols_sum = cols_sum + skills_selected.iloc[:,i]
  skills_selected["select_col"] = cols_sum == 0
  skills_["select_col"] = skills_selected["select_col"]
  st.write(f"The jobs with the lowest AI Impact, based on the skill(s) you selected:")
  st.skillsdf=st.dataframe(
   (skills_[(skills_["cluster"]==cluster_skills) & (skills_["select_col"] == False)][["Title", "rank_new","AI Impact"]].sort_values(by="rank_new").head(3).set_index("Title")).style.set_properties(**{"background-color": "white", "color": "black"}).format({"rank_new": "{:.2f}"})
)
  skills_.drop(columns=["select_col"])