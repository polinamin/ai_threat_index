

# Capstone: Predicting AI Threat on individual Jobs

---

### Overview
The impact of Artificial Intelligence on employment prospects is a well-debated topic, but there are currently no reliable models to predict AI impact by occupation. Here, we seek to develop a model and a tool that would help job-seekers proactively upskill and optimize their job search to withstand the potential impacts of AI replacement.

The following steps were followed, in order to complete this analysis:
1. Use [O\*NET occupation data, US Census Bureau](https://www.onetcenter.org/) to collect occupation-level data on the following:
* Title (cross-data field, utilized for dataset merging): Altogether, there were 853 job titles in the data. However, some datasets had more occupations that were evaluated for that specific parameter. When doing individual analyses, all available data was utilized. 
     * Of the 853 occupations in our dataset, we completed the analysis on 757 rows of data (where a reasonably reliable match existed between the two dataset sources.)
* [Industry](https://www.census.gov/topics/employment/industry-occupation/guidance/code-lists.html): There were 21 distinct industries in the dataset. Data mapped Job Titles to Industries, including a field which consolidated all industries where that occupation reported functioning. Only the dominant industry was utilized for the analysis (after comparing this to the joint industry and the stacked industry approaches.)
* [Task](https://www.onetcenter.org/dictionary/28.0/excel/task_ratings.html): There were 161,847 distinct tasks included in this dataset, which was organized by frequency of use (1 being the least frequent (annually), 5 being the most frequent (more than hourly)). For this analysis, only the words in the task statements were utilized (as tasks did not overlap and/or align by occupation.)
* [Tools](https://www.onetcenter.org/dictionary/28.0/excel/technology_skills.html): 8,743 tools were included in this dataset, indicating the tools an occupation reported needing to fulfill the responsibilities of their job.
* [Skills](https://www.onetcenter.org/dictionary/28.0/excel/skills.html) - 35 individual skills associated with each occupation, including a measure of the importance of the feature to each occupation and the level of proficiency required. This data was converted into a single combined feature of (importance * level) and then used as a scaled feature matrix in further analysis.
<br>These were individually processed, analyzed and modeled, to understand whether individual occupation features play a role in predicting AI Impact.


2. Use [AI Job Threat Index](https://www.kaggle.com/datasets/manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index) to get an independent assessment of AI threat.
    * The dataset consisted of 4,706 job titles and AI Impact scores. A [fuzzy match](https://pypi.org/project/thefuzz/#description) algorithm was utilized in order to complete a partial match on the data. While various matching approaches were considered, the method yielding the highest aggregate total of matched titles across the datasets.
    * This was categorized on a scale of 1-4 to complete a multiclassification model and determine AI Impact for each occupation in the dataset.

3. Data was split using train-test-split, first separately and then together, in order to gauge predictive strength for the model. All data runs were completed via the LogisticRegression model, varying NLP, preprocessing and regularization techniques to improve model performance.
    * The baseline for the model was as follows: 

| **AI Impact Category** | **% Distribution** | 
| --- | --- | 
| Very High Impact | 2.9% | 
| Very High Impact | 13.5% | 
| High Impact | 35.1% | 
| Moderate Impact | 48.5% | 


* While most of the models performed better than 48.5%, there was significant overfitting and underperformance on the test dataset. All this signals the need for:
- Additional data to increase our sample size
- Additional features to better capture the impact of AI on a given occupation


### Modeling

| **Job Parameter** | **Approach** | **Model** | **Train** | **Test** | 
| --- | --- | --- | --- | --- | 
| Title | NLP | Logistic Regression - CVEC | 0.716 | 0.542 | 
| Title | NLP | Logistic Regression - TFIDF  | 0.61 | 0.516 | 
| Task |  NLP | Logistic Regression - CVEC | 0.76 | 0.484 | 
| Task | NLP | Logistic Regression - TFIDF | 00.79 | 0.495 | 
| Tool | NLP | Logistic Regression - CVEC | 0.642 | 0.447 | 
| Tool | NLP | Logistic Regression - TFIDF | 0.614 | 0.458 | 
| Tool | Binary | Logistic Regression  | 0.6 | 0.505 |
| Industry | Binary with dominant Industry | Logistic Regression | 0.543 | 0.521 | 
| Industry | Binary with all Industries | Logistic Regression  | 0.479 | 0.431 |
| Industry | NLP | Logistic Regression - CVEC  | 0.524 | 0.328 |
| Skills | Scaled | Logistic Regression | 0.578 | 0.605 |


**Combined Analysis**

 **Approach** | **Model** | **Train** | **Test** |
| --- | --- | --- | --- |
| Categorical (Tool, Industry), Scaled (Skill) + NLP (Task, Title) | Logistic Regression - CVEC | 1.0 | 0.537 |
| Scaled(Skill) + NLP (Task, Title, Tool, Industry) | Logistic Regression - TFIDF | 0.945 | 0.542 |
| Scaled(Skill) + NLP (Task, Title, Tool, Industry) | Gridsearch (Logistic Regression - CVEC) | 0.707 | 0.532 | 
| Scaled(Skill) + NLP (Task, Title, Tool, Industry)| Gridsearch (Logistic Regression - TFIDF) | 0.64 | 0.5|
| Scaled(Skill) + NLP (Task, Title, Tool, Industry)| Multinomial | 0.981 | 0.505|


### Model Selection and Evaluation
While no model performed well, all of them slightly improved on the baseline of the target variable (48%). The model that was selected was **Logistic Regression with a CountVectorizer, Standard Scaler and L2 (Ridge) Regularization**. 

The final parameters were:

'max_features' : [5000]<br>
'min_df' : [2]<br>
'max_df' : [0.9]<br>
'ngram_range' : [(1, 2)]<br>
'top_words' : [stops]<br>
'C':[0.001]<br>
'penalty': ['l2']<br>
'max_iter': [3000]<br>

### Next steps would require:
* A thorough analysis of literature around job classification and variables sensitive to AI
* Additional data to validate or substitute current AI Impact data, in order to gain more confidence in the target variable
* A larger sample size (with more hypertuning with the Fuzzy Matching process, and additional occupations for consideration
* A qualitative assessment of AI vulnerability, to create data weights on model features
* A dynamice integration with Litix (LinkedIn API) to apply model outcomes toward a more optimized and strategic job search, learning and development strategy for employee upskilling, people strategy consultation for a more forward-looking workforce planning strategy, career growth and mentorship program development, and internal mobility consultation around transferability of skills within organizations.
