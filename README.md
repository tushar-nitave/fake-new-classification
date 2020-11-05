## fake-new-classification

#### Done
1. Exploratory Data Analysis
    - dataset size
    - check if imbalanced
    - missing data
    - frequency of each class 
2. Preprocessing
    - remove stopwords
    - stemming
    - lowercase and remove punctuation
---
#### To Do
3. Feature Engineering
    - using word2vec
4. Model Architecture
    - CNN
    - full connected 
5. Train and Validate
    - hyperparameter tuning
6. Test
7. Evaluation Metrics
    - Confusion matrix
    - F1 score
    - Precision
    - Recall
    - ROC
8. Deploy
    - Flask server
9. Front-end (optional)
---
#### Findings

It is observed that we are dealing with highly imbalanced data so we have to use different strategies for 
sampling the data to make it balanced.

Make dataset balanced:
    - SMOTE
    - undersampling
    - SMOTE + ENN