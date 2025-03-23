# TABLE OF CONTENTS
Python projects created while at the Triple Ten Data Science bootcamp, by Deborah Thomas

## Sprint17_Interconnect_Churn_project
Build a model, for Interconnect phone company, that predicts which customers will churn.  The evaluation metric is AUC-ROC, and must have a score greater than or equal to 0.75.
##### Libraries used:
- **NumPy**: For numerical calculations and handling large arrays efficiently.
- **Pandas**: Essential for data analysis and preprocessing of datasets.
- **Matplotlib**: For data visualization to create informative plots.
- **Seaborn**: Enhances visualizations and provides a high-level interface for drawing attractive graphics.
- **Scikit-learn**: Core library for machine learning that includes various algorithms and tools for model evaluation.
- **Statsmodels**: For statistical modeling and hypothesis testing.
- **Keras/TensorFlow**: Frameworks for building and training neural network models

##### Models used:
- **Random Forest Classifier**: 
- **Gradient Boosting Classifier**: 
- **Decision Tree Classifier**: 
- **Support Vector Classifier (SVC)**:
- **Logistic Regression**: 

##### Link to open project:
  - [Sprint17_Interconnect_Churn_project](https://github.com/Script-Whiz/Sprint17_Interconnect_Churn/blob/0f1208183f323112ee2305210d1bb061a3c5d2c9/notebooks/Sprint17_Interconnect_Churn_ver3.ipynb)

## Sprint15_CNN_agePrediction
Looks at images, and predicts age of customers, to help Good Seed Company adhere to alcohol laws, by not selling to customers under the age of 21.
##### Libraries used:
- **Pandas**: For numerical calculations and handling large arrays efficiently.
- **OS**: Essential for operating system-related tasks.
- **Pillow (PIL)**: For image processing and manipulation.
- **Random**: For generating random numbers and selecting random items.
- **TensorFlow**: For building and training machine learning models.
- **Matplotlib**: For data visualization to create informative plots.
- **Inspect**: For getting information about live objects, including modules and classes.
- **ImageDataGenerator**: For augmenting image data in real-time during training (from TensorFlow Keras Preprocessing).
- **ResNet50**: A deep learning model for image classification (from TensorFlow Keras Applications).
- **Sequential Model**: For creating linear stacks of layers in a neural network (from TensorFlow Keras Models).
- **Layers**: Including GlobalAveragePooling2D, MaxPooling2D, Dense, Dropout, Flatten, Conv2D (from TensorFlow Keras Layers).
- **Adam Optimizer**: For optimizing neural network training (from TensorFlow Keras Optimizers)

##### Models used:
- **Random Forest Classifier**:
- **Gradient Boosting Classifier**:
- **Decision Tree Classifier**:
- **Support Vector Classifier (SVC)**:
- **Logistic Regression**:

#####  Link to open project:
  - [Sprint15_CNN_agePrediction](https://github.com/Script-Whiz/Sprint15_CNN_agePrediction/blob/d9a305eb2b8b57f4a291af0187024f25ea013340/notebooks/Sprint15_CNN_agePrediction_final_ver.ipynb)

## Sprint14_Chapter2_Vectorization
Train a model to learn  whether or not a review is positive, or negative, for the Film Junky Reviews Company. Model F1 score must be better than 0.85.
##### Libraries used:
- **Math**: For mathematical operations and functions.
- **NLTK**: For natural language processing tasks and accessing stopwords.
- **NumPy**: For numerical computations and array operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For creating static, animated, and interactive visualizations.
- **Seaborn**: For statistical data visualization based on matplotlib.
- **TQDM**: For adding progress bars to loops and iterations.
- **Scikit-learn**: For machine learning tools, including metrics, dummy classifiers, and feature extraction.
- **Spacy**: For advanced natural language processing tasks.
- **LightGBM**: For gradient boosting framework using tree-based learning algorithms.
- **PyTorch**: For deep learning and neural networks.
- **Transformers**: For state-of-the-art natural language processing models

##### Models Used:
- **Dummy Classifier**: For baseline model performance.
- **Logistic Regression**: For binary classification tasks.
- **LightGBM Classifier**: For gradient boosting classification.
- **Transformer Models**: For advanced NLP tasks (via the transformers library)

##### Link to open project:
  - [Sprint14_Chapter2_Vectorization](https://github.com/Script-Whiz/Sprint14_Chapter2_TextVectorization/blob/29f3afd8bd4001be2add2f2b11232011e5adbdb3/notebooks/Sprint14_Chapter2_TextVectorization_ver2_final.ipynb)

## Sprint13_SweetTaxi_timeSeries
A time series project that predicts how many customers the Sweet Taxi Company will have during the next hour.

##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Plotly Express**: For interactive data visualizations.
- **Statsmodels**: For statistical computations, time series analysis, and plotting.
- **Matplotlib**: For data visualization and plotting.
- **Scikit-learn**: For machine learning implementations and metrics.
- **XGBoost**: For gradient boosting framework.
- **pmdarima**: For automatic ARIMA model fitting.

##### Models Used:
- **Linear Regression**: For basic linear modeling.
- **Decision Tree Regressor**: For non-linear regression using decision trees.
- **Random Forest Regressor**: For ensemble learning using multiple decision trees.
- **Gradient Boosting Regressor**: For boosting-based regression.
- **XGBoost**: For extreme gradient boosting regression.
- **ARIMA**: For time series forecasting

##### Link to open project:
  - [Sprint13_SweetTaxi_timeSeries](https://github.com/Script-Whiz/Sprint13_SweetTaxi_timeSeries/blob/main/notebooks/Sprint13_SweetTaxi_timeSeries_ver5_final.ipynb)

## Sprint12_RustyCar_GradBoost
Supervised Learning project, to predict car prices for the Rusty Car Company. RMSE is used for scoring the model.

##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Scikit-learn**: For various machine learning tasks, including model training and evaluation.
- **Matplotlib**: For data visualization.

##### Models Used:
- **Linear Regression**: For basic linear modeling.
- **Decision Tree Regressor**: For regression using decision trees.
- **Random Forest Regressor**: For ensemble regression using multiple decision trees.
- **Gradient Boosting Regressor**: Implemented via `LGBMRegressor` (LightGBM).
- **CatBoost Regressor**: For gradient boosting with categorical features
- **XGB Regressor**: For extreme gradient boosting regression

##### Link to open project:
  - [Sprint12_RustyCar_GradBoost](https://github.com/Script-Whiz/Sprint12_RustyCar_GradBoost/blob/main/notebooks/Sprint12_RustyCar_GradBoost_final.ipynb)

## Sprint11_Ins_US_matrix
This project answers 4 questions:

- Task 1: Find customers who are similar to a given customer.
- Task 2: Predict whether a new customer is likely to receive an insurance benefit.
- Task 3: Predict the number of insurance benefits a new customer is likely to receive using a linear regression model. 
- Task 4: Protect clients' personal data, using masking, or data obfuscation.

##### Libraries used:

- **NumPy**: For numerical computations and array operations.
- **Pandas**: For data manipulation and analysis.
- **Math**: For mathematical functions.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For various machine learning tasks including linear models, metrics, and preprocessing.
- **Plotly Express**: For interactive data visualizations.
- **Streamlit**: For creating web applications for machine learning and data science projects.
- **PIL (Pillow)**: For image processing.
- **IPython**: For interactive computing environment (specifically for displaying outputs).

##### Models Used:
- **Linear Regression**: For basic linear modeling.
- **K-Nearest Neighbors Classifier**: For classification tasks based on nearest neighbors.
- **Nearest Neighbors**: For unsupervised learning tasks like density estimation and anomaly detection.

##### Link to open project:
  - [Sprint11_Ins_US_matrix](https://github.com/Script-Whiz/Sprint11_Ins_US_matrix/blob/36aa96674fe6235fff387811f8fabedb3fb9cc4f/notebooks/Sprint11_LinearAlg_ver2_final.ipynb)

## Sprint10_Gold_Recovery
Studies the gold purification process. Uses predictive models are built to predict gold levels for the final purification stage.

##### Libaries used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Matplotlib**: For data visualization.
- **IPython**: For interactive computing (used for displaying outputs in Jupyter).
- **Scikit-learn**: For machine learning tasks.

##### Models Used:
- **Linear Regression**: For basic linear modeling.
- **Decision Tree Regressor**: For regression using decision trees.

##### Link to open project:
  - [Sprint10_Gold_Recovery](https://github.com/Script-Whiz/Sprint10_Gold_Recovery/blob/main/notebooks/Gold_Recovery_ver6_final.ipynb)

## Sprint9_OG_Mining_MLB
Analyze ore refinery process for Oily Giant Company. Find profit margin, revenue prediction for best site for digging new well, profitability threshhold, probability of loss. Bootstrapping used. Multiple datasets used.

##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Scikit-learn**: For machine learning tasks, including model training and evaluation.
- **IPython**: For interactive computing (used for displaying outputs in Jupyter).
- **Matplotlib**: For data visualization.
- **Plotly**: For interactive data visualizations (using `graph_objects` and `io`).
- **Termcolor**: For colored terminal text outputs.

##### Models Used:
- **Linear Regression**: For basic linear modeling

##### Link to open project:
  - [Sprint9_OG_Mining_MLB](https://github.com/Script-Whiz/Sprint9_OG_Mining_MLB/blob/main/notebooks/OG_mining_ver5_final.ipynb)

## Sprint8_Churn_Beta
Discovering churn rates of customers using supervised learning, One hot encoding, EDA, data visualization using different types of graphs, balancing of data using downsampling. Scoring, using Accuracy, Precision, Recall, F1 score, and AUC-ROC.

##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and operations.
- **Scikit-learn**: For various machine learning tasks including model training and evaluation.
- **Plotly Express**: For interactive data visualizations.

##### Models Used:
- **Decision Tree Classifier**: For classification using decision tree algorithms.
- **Logistic Regression**: For binary classification problems.
- **Random Forest Classifier**: For ensemble classification using multiple decision trees

##### Link to open project:
  - [Sprint8_Churn_Beta](https://github.com/Script-Whiz/Sprint8_Churn_Beta/blob/main/notebooks/Sprint8_BetaBank_Churn_ver6.ipynb)

## Sprint7_Intro_MachineLearning
Supervised Learning, and hypertuning of models for MegaLine Phone Company.  Using EDA and Data Visualization to find which phone plan is best: Smart or Ulta     Scoring used: Accuracy

##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning tasks including training, metrics, and model selection.
- **Plotly Express**: For interactive data visualizations.

##### Models Used:
- **Decision Tree Classifier**: For classification tasks using a decision tree approach.
- **Logistic Regression**: For binary classification problems.
- **Random Forest Classifier**: For ensemble classification using multiple decision trees.

##### Link to open project:
  - [Sprint7_Intro_MachineLearning](https://github.com/Script-Whiz/Sprint7_Intro_MachineLearning/blob/main/notebooks/Sprint7_Intro_MachineLearning_version8_final.ipynb)

## Sprint6_Zuber_rides_time
Time Series analysis of Zuber rides. Merging databases, EDA, hypothesis testing. Looking at data, from competitors, and testing a hypothesis about the impact of weather on ride frequency, identifying the top 10 neighborhoods, in terms of drop-offs, and making conclusions from data visualization.

##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **Plotly Express**: For interactive data visualizations.
- **SciPy Stats**: For statistical functions and distributions

##### Link to open project:
  - [Sprint6_Zuber_rides_time](https://github.com/Script-Whiz/Sprint6_Zuber_rides_time/blob/main/notebooks/Sprint6_Zuber_rides_time_ver4_final.ipynb)

## Sprint5_VideoGames_imputation
Analysis of historic video game sales, from 1980-2016, from the (fictitional) online store called "Ice". This dataset includes sales from North America, Europe, Japan. Sales analysis from those 3 regions, to gain an understanding of which video games will be successful. EDA, data cleaning using imputation.
 
##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and handling arrays.
- **Plotly Express**: For interactive data visualizations.
- **Seaborn**: For statistical data visualization.
- **Matplotlib.pyplot**: For creating static, animated, and interactive visualizations in Python.
- **SciPy Stats**: For statistical functions and distributions.
- **Plotly Subplots**: For creating subplots in Plotly.
- **Random**: For generating random numbers and selections
 
##### Link to open project:
  - [Sprint5_VideoGames_imputation](https://github.com/Script-Whiz/Sprint5_VideoGames_imputation/blob/fc14a3bba74cba808437ce1500ad9e377182d6b4/notebook/games_ver_16.ipynb)


## Sprint4_Auto_Comparison
Comparison of auto sales from 1908-2019, using EDA and data visualization, looking for average auto sales prices, and correlations between auto attributes and sales. Virtual environments were set up for this.

##### Libraries Used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and handling arrays.
- **Plotly Express**: For interactive data visualizations.
- **Seaborn**: For statistical data visualization.
- **Matplotlib.pyplot**: For creating static, animated, and interactive visualizations in Python.
- **SciPy Stats**: For statistical functions and distributions.
- **Plotly Subplots**: For creating subplots in Plotly.
- **Random**: For generating random numbers and selections

##### Link to open project:
  - [Sprint4_Auto_Comparison](https://github.com/Script-Whiz/Sprint4_Auto_Comparison/blob/b06dd84fc9e3174ab038488f44378126705ccf79/notebooks/Sprint4_Auto_Comparison_final.ipynb)

## Sprint3_Megaline_Phone_revenue
Data cleaning and EDA to analyze customers habits for Megaline phone company, and find which phone plan brings in more revenue..
##### Libraries used:
- **Pandas**: For data manipulation and analysis.
- **IPython.display**: For displaying HTML and other representations in Jupyter notebooks.
- **NumPy**: For numerical computations and handling arrays.
- **Matplotlib**: For creating static plots and visualizations.
- **Seaborn**: For statistical data visualization.
- **Warnings**: To manage warning messages in your code.
- **Cycler**: For creating color cycles for plots.
- **SciPy Stats**: For statistical functions and distributions

##### Link to open project:
  - [Sprint3_Megaline_Phone_revenue(https://github.com/Script-Whiz/Sprint3_MegalinePhone_revenue/blob/54b720c224ce131da31b00bb94becff36fa52396/notebooks/.ipynb_checkpoints/Megaline_final_ver1-checkpoint.ipynb)

## Sprint2_Instacart_EDA
Use EDA to analyze Instacart ordering habits. Find what are the top 20 items ordered most frequently?
##### Libraries used:
- **Pandas**: For data manipulation and analysis
- **Matplotlib**: For creating static plots and visualizations.
- **IPython.display**: For displaying HTML and other representations in Jupyter notebooks

##### Link to open project:
  - [Sprint2_Instacart_EDA](https://github.com/Script-Whiz/Sprint2_Instacart_EDA/blob/cb8e157544942a87fd65db77dbaaff7fb2be2709/notebooks/Sprint2_Instacart_EDA_ver2.ipynb)

## Sprint1_DataCleaning
Data cleaning, and basic EDA, on the Golden Age of movies,  to discover how the number of votes a movie receives affects its ratings.
##### Libraries used:
- **Pandas**: For data manipulation and analysis

##### Link to open project:
  - [Sprint1_DataCleaning](https://github.com/Script-Whiz/Sprint1_DataCleaning/blob/4c0782a1f41bf976e489159b3de7dfb3049ffd6c/notebooks/Sprint1_DataCleaning_ver2.ipynb)

