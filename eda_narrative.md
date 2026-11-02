# Exploratory Data Analysis (EDA) Narrative

## 1. What the Data Shows
The analyzed dataset contains passenger records with a mix of numerical and categorical features such as Age, Fare, Pclass, Sex, and Embarked. Distribution plots reveal that passenger ages are moderately right-skewed with a peak among young adults, while fare prices show a heavy concentration of lower-cost tickets with extreme high-value outliers. Category counts highlight distinct groupings across classes and embarkation points, establishing foundational demographic patterns.

## 2. What is Suspicious
Several data anomalies are immediately apparent. The Age column contains 177 missing values, and the Cabin column has a critical deficit with 687 missing entries out of the total rows. Additionally, extreme outliers in fare prices and sparse categorical distributions indicate potential noise or skewed collection metrics that could distort predictive performance if left unaddressed.

## 3. What Needs Fixing
To prepare this dataset for machine learning models, missing values in numerical columns like Age require robust imputation strategies (such as median filling), while columns with excessive missing data like Cabin require evaluation for dropping or binary transformation. Outlier clipping and proper encoding of categorical features are essential next steps before modeling.