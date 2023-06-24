# Depression Detection on Social Network with Natural Language Processing
![Visual Studio Code](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Python](https://img.shields.io/badge/Python-F6D049?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-5BB0DF?style=for-the-badge&logo=sqlite&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly_Dash-3D4D71?style=for-the-badge&logo=plotly&logoColor=white)

## Description

This project was carried out to introduce an NLP-based architecture for accurately predicting depressive risks using social media posts and identifying depressive symptoms associated with the posts. It utilized various techniques to enable prediction models to distinguish between depressive and non-depressive posts. A new technique was proposed to demonstrate the possibility of using a custom LIWC dictionary to detect depressive symptoms from text. The analysis results were visualized on a web-based dashboard application. This application can assist mental health professionals in identifying behavioral signs of depression and alerting them to patients who may be at risk. Additionally, patients can utilize the system to assess their mental health status and contact their doctors as needed. Since anyone can use it to check their own mental health, the system has the potential to enhance public awareness about mental health issues.

The underlying motivation behind this project was to address the growing concern of mental health during challenging times such as the Covid-19 pandemic. I noticed a significant increase in negative posts on social media platforms, which indicated a potential rise in mental health issues among individuals. I wanted to contribute to raising awareness about mental health issues and provide a platform that could offer support and guidance to those in need. By harnessing the potential of NLP and developing a user-friendly web-based application, I aspired to make a positive impact on the lives of individuals struggling with mental health challenges, while also promoting overall mental well-being in society.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Data Collection](#data-collection)
- [Data Screening](#data-screening)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Final Dataset](#final-dataset)
- [Classification Model](#classification-model)
- [System Diagrams](#system-diagrams)
- [System Screenshots](#system-screenshots)

## Installation

```bash
python -m venv venv
```

## Usage

```bash
cd venv/scripts
activate.bat
cd ..
python app.py
```

## System Architecture

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/System%20Architecture.png?raw=true" width="500">

The data mining steps were uploaded to /Data Analysis folder for easier exploration using Jupyter Notebook. Besides, images included in this file can also be accessed from /docs folder.

## Data Collection

The dataset was collected using Twint, an advanced Twitter scraper that retrieves tweets without using Twitter’s API. Two datasets were required, a normal (non-depressed) dataset and a depressed dataset.

#### Normal Dataset (Non-Depressed)

The normal dataset was collected in 4 batches, each spanning a single day, from 16th to 18th December 2021, and on 31st December 2021. A total of 3000 tweets were collected per day, except for the last day when only 1000 tweets were collected. In total, the dataset comprised 10,000 tweets. Collecting the dataset on different days aimed to avoid similar topics being discussed on the same day and to enhance the diversity of topics within the normal dataset.

#### Depressed Dataset

Various keywords were selected to target depressive tweets, including different hashtags related to depression, such as “depressed”, “depression”, “anxiety”, “bipolar”, “DepressionIsReal”, and more. Depressed candidates were manually selected from the dataset gathered using these hashtags. Their profiles were scrutinized to ensure they exhibited depressive tendencies, such as expressing thoughts of suicide, loneliness, and self-hatred.

## Data Screening

In this step, irrelevant tweets were manually removed using the following predefined rules. Once this process was completed, an additional column named “target” was assigned the value “normal”, while in the depressed dataset, it was assigned the value “depressed”.

#### Normal Dataset

*	Remove non-English tweets. 
*	Remove tweets discussing duplicate topics, such as NFTs, giveaways, and birthday wishes.
*	Remove tweets with commercial intent, often originating from business accounts.
*	Remove tweets sharing songs, which usually consisted of the song title and an external link.
*	Remove tweets displaying depressive tendencies.

#### Depressed Dataset

*	Remove non-English tweets.
*	Remove tweets with motivational content.
*	Remove tweets without any depressive tendencies.

## Data Preprocessing

The tweet column was preprocessed to clean and prepare the data for analysis. The processed tweets were then added back to the dataset, allowing for easy comparison and analysis alongside the original tweets.

*	Extract 4 required columns, including datetime, username, tweet, and target.
*	Remove duplicate tweets, non-English characters, and stop words.
*	Remove punctuations, numbers, and whitespaces.
*	Remove links, emails, mentions, hashtags, Unicode, and special characters.
*	Convert emoji to text and text to lowercase.
*	Expand contractions and jargons.
*	Fix word lengthening.
*	Lemmatizing.
  
## Feature Engineering

4 techniques were applied to extract useful linguistic features that can effectively describe and differentiate depressive and non-depressive tweets. After feature engineering process, the dataset now consists of 10547 rows and 30 columns. 

#### Valence Aware Dictionary & sEntiment Reasoner (VADER)

VADER is capable of detecting sentiment polarity (positive, neutral, or negative) within a text. It also tells the intensity of emotion by considering capitalization and punctuation. The original tweets were selected for applying VADER instead of the cleaned tweets as they preserve the intensity of emotion. Only the "compound" score was used as it provides a summary of the sentiment.

#### National Research Council Canada Affect Lexicon (NRCLex)

NRCLex can extract 10 emotional affects behind the text, including fear, anger, anticipation, trust, surprise, positive, negative, sadness, disgust, and joy. Since NRCLex requires text cleaning to ensure accurate analysis, thus the cleaned tweets were chosen as input for calculating the emotion lexicon.

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/NRCLex.png" width="500">

#### Linguistic Inquiry and Word Count (LIWC)

LIWC is a text analysis program that counts words in over 80 categories with psychologically meaning. The dictionary plays an crucial role in the LIWC program as it specifies which words should be included in the analysis. For this project, a customized version of the dictionary was created, taking inspiration from the original dictionary, to identify depressive symptoms within tweets. The full dictionary used can be accessed from Data Analysis/liwc_custom.xlsx.

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Custom%20LIWC.png" height="300">

The inclusion of the first two categories of personal pronouns was based on the observation that depressed people tend to use first person pronouns, while non-depressed people tend to use second or third person pronouns. Categories 3 to 9 and 13 represent depressive symptoms, and categories 10 to 12 represent psychological stressors. Psychological stressors refer to the social and physical environmental circumstances that affect one’s mental health. The dictionary contains only 142 words as it was manually compiled from a limited number of papers and resources on this topic.

#### Gibbs Sampling Dirichlet Multinomial Mixture (GSDMM)

GSDMM is a modified version of Latent Dirichlet Allocation (LDA) short text clustering model. It is specifically designed to be more effective in detecting topics within shorter texts, such as tweets. The GSDMM model was initialized with parameters K=10, alpha=0.1, beta=0.1, and n_iter=30. Tweets are assigned a specific topic only if the threshold equals or higher than 0.3. Otherwise, the tweet is assigned to topic 10, indicating another topic.

## Final Dataset

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Final%20Dataset1.png" width="700">
<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Final%20Dataset2.png" width="700">

## Classification Model

#### Machine Learning Models

* Logistc Regression (LR)
* Support Vector Machine (SVM)
* Random Forest (RF)
* Neural Network (NN)

#### Word Embeddings

* Term Frequency-Inverse Document Frequency (TF-IDF)
* Word2Vec
* Bidirectional Encoder Representations from Transformers (BERT)

#### Training Input

* Cleaned tweets only
* All features

#### Combinations

A total of 8 models were constructed using various combinations of machine learning models, word embeddings techniques, and training inputs. The performance of each model was evaluated using 10-fold cross-validation to calculate the mean performance. To generate training samples, the dataset was shuffled, and a selection of 5000 samples from both normal and depressed tweets was made. From these samples, four columns containing username, datetime, original tweet, and target were excluded. The resulting sample was split into 80% training data and 20% testing data respectively for model training and evaluation.

* LR with TF-IDF (tweets only)
* LR with TF-IDF (tweets + feature extracted from feature engineering)

* SVM with TF-IDF (all features)
* RF with TF-IDF (all features)

* SVM with Word2Vec (tweets only)
* SVM with Word2Vec (all features)

* NN with BERT (tweets only)
* NN with BERT (all features) 93.90%

#### Model Evaluation

The results of the study indicate that including all linguistic features in the training process yields better performance compared to using only tweets. When comparing the performance of three machine learning models using the same embeddings and features, SVM exhibited the highest performance, while RF had the lowest performance. RF was found to be the least efficient model in terms of training time. To improve the SVM model's performance, it was retrained using a more complex word embedding technique, Word2Vec, which resulted in further improvement. Additionally, the neural network model demonstrated the highest accuracy, achieving a score of 93.90%, indicating its effectiveness in enhancing performance.

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Model%20Evaluation.png" width="500">

## System Diagrams

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Use%20Case%20Diagram.png" width="600">
<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Activity%20Diagram.png" width="600">

## System Screenshots

This section provides examples of the dashboard visualizations generated by the system. For a comprehensive demonstration of the entire system, you can access the full video from [here](https://drive.google.com/file/d/1NPo5IM3dPUtsm6i6y9K_zDpyFJ2GL0ZO/view?usp=sharing).

<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Screenshot1.png" width="700">
<img src="https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/Screenshot3.png" width="700">

The dashboard layout includes a table displaying information about the tweets, including the datetime, tweet, symptoms, and target. Depressive tweets are highlighted in red for easy identification. The depression score, calculated by dividing the total number of depressed tweets by the total number of tweets, provides an overall measure of the patient's depressive tendencies.

To track the patient's emotional state over time, a time series graph illustrates the sentiment changes. This allows doctors to observe patterns and fluctuations in the patient's emotions. Additionally, positive and negative word clouds offer a comprehensive overview of the patient's experiences. For example, the negative word cloud may highlight words like "rest" and "sleep," indicating sleep disturbances and a lack of rest.

To provide an immediate assessment, an alert is displayed at the top of the dashboard based on the depression score range. This helps doctors quickly identify the severity level of the patient's depression, with different ranges indicating lower, middle, or higher levels of depression.

