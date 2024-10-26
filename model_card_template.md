# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Wei Liu created this model. 
- This Random Forest model, developed with Scikit-learn, performs binary classification.
- It is implemented as part of the Udacity Machine Learning DevOps Nanodegree, based on a foundational structure provided by Udacity.

## Intended Use
- The model should be used to predict whether a person’s salary is <=50K or >50K. 

## Training Data
- The model was trained on data from the [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income).
- 80% of the dataset was allocated for model training.

## Evaluation Data
- 20% of the dataset was reserved for evaluating model performance.

## Metrics
- Precision: 0.75
- Recall: 0.62
- F1: 0.68

## Ethical Considerations
- Evaluating potential biases is essential before applying this model to other use cases. We performed a slice analysis focused on features like workclass, education, and native country.
- Reviewing these metrics is key to identifying and mitigating biases, promoting fair and ethical model usage.

## Caveats and Recommendations
- This model is foundational and can be enhanced with further refinement. Improvements could include more sophisticated preprocessing techniques and thorough hyperparameter tuning.
- For practical applications, using up-to-date datasets is recommended to maintain the model's relevance and accuracy.