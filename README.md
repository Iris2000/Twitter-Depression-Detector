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

![System Architecture](https://github.com/Iris2000/Twitter-Depression-Detector/blob/master/docs/System%20Architecture.png?raw=true)

The data mining steps were uploaded to /Data Analysis folder for easier exploration using Jupyter Notebook.

## Data Collection

The dataset was collected using Twint, an advanced Twitter scraper that scrapes tweets without using Twitter’s API. Two datasets were required, a normal (non-depressed) dataset and a depressed dataset.

#### Normal Dataset (Non-Depressed)

The normal dataset was collected in 4 batches of one day each from 16th to 18th December 2021, and 31st December 2021. 3000 tweets are collected per day, only 1000 tweets on the last day. It has 10,000 tweets in total. The dataset was collected from different days to avoid similar topics discussed on the same day, and to augment the normal dataset with topics diversity.

#### Depressed Dataset

A range of keywords were chosen to target depressive tweets, including different hashtags related to depression, such as “depressed”, “depression”, “anxiety”, “bipolar”, “DepressionIsReal”, and more. Depressed candidates were manually selected from the dataset collected using these hashtags. Their profiles were scrutinized to make sure they showed depressive tendencies such as suicidal, loneliness, and self-hatred.

## Data Screening

In this step, irrelevant tweets from the dataset were manually removed using different rules listed below. When done, an additional column called “target” was added to each dataset. The “target” column in the normal dataset was set to “normal”, while the depressed dataset was set to “depressed”.

#### Normal Dataset

*	Remove non-English tweets. 
*	Remove tweets with duplicate topics, such as NFTs, giveaways, and birthday wishes.
*	Remove tweets with commercial purposes, typically from business accounts.
*	Remove tweets that share songs, which usually contain only the song title and an external link direct to the song.
*	Remove tweets with depressive tendencies.

#### Depressed Dataset

*	Remove non-English tweets.
*	Remove tweets with motivational purposes.
*	Remove tweets without depressive tendencies.

## Data Preprocessing

Data preprocessing was done on the tweet column. The processed tweet column was then combined to the dataset so that it has both the original and cleaned tweets. 

*	Extract 4 required columns, including datetime, username, tweet, and target.
*	Remove duplicate tweets, non-English characters, and stop words.
*	Remove punctuations, numbers, and whitespaces.
*	Remove links, emails, mentions, hashtags, Unicode, and special characters.
*	Convert emoji to text and text to lowercase.
*	Expand contractions and jargons.
*	Fix word lengthening.
*	Lemmatizing.
  
## Feature Engineering

4 techniques were applied to extract useful linguistic features that are capable of describing and distinguishing depressive and non-depressive tweets. After feature engineering, the dataset has a total of 10547 rows and 30 columns. 

#### Valence Aware Dictionary & sEntiment Reasoner (VADER)

VADER can detect sentiment polarity within a text, as either positive, neutral, or negative. It also tells intensity of emotion by considering capitalization and punctuation. The original tweets were chosen to apply VADER rather than cleaned tweets because original tweets retain the intensity of emotion. Only "compound" was returned because it represents the summary of the score.

#### National Research Council Canada Affect Lexicon (NRCLex)


## Features

If your project has a lot of features, list them here.
