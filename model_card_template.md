# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used for this project is RandomForestClassifier from sklearn.model.RandomForestClassifier. This model is created by Sreevidya Krishnaswamybhattar. 

## Intended Use

Using the census data, the model predicts whether a person earns more than 50k or not. 

## Training Data

More information about the census data can be found in this here: https://archive.ics.uci.edu/dataset/20/census+income. The extraction was done by Barry Becker from the 1994 Census database. 

Features:

- age: Integer
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: Integer
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: Integer
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: Integer
- capital-loss: Integer
- hours-per-week: Integer
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Evaluation Data

The raw dataset is first cleaned and processed. The cleaned dataset is split into training and evaluation data with is size of 20%. 

## Metrics

The metrics the model used are Precision, Recall, and F1
Performance of the model:
- Precision: 0.7296
- Recall: 0.6288
- F1: 0.6754

## Ethical Considerations

The model uses census data and is not biased toward any particular group of people.

## Caveats and Recommendations

The Random Forest Classifier model are better suitable for batch prediction. This model might take longer prediction times for On-the-fly predictions, so it is not suitable for real-time prediction.