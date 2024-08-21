# MINORPROJECT
Let's walk through the process step by step and clarify the goal, methods, and current implementation in our project titled **"Predicting Software Bugs Using Machine Learning Techniques."**

### Project Goal
The primary objective of your project is to **predict the likelihood or severity of software bugs** using various machine learning models. By training models on a dataset that includes information about past software bugs, we aim to make predictions about new or unseen bugs based on their descriptions and related features.

### Overview of the Process
1. **Data Collection:**
   - **Datasets:** You provided several datasets in CSV format (like `fix.csv`, `sev.csv`, etc.), a text corpus (`corpus(fixsev).txt`), and embeddings in `.npy` format.
   - **Text Data:** This includes bug descriptions, which are often unstructured text.
   - **Categorical Data:** This includes features like `Severity` that need to be encoded before use in models.

2. **Data Preparation:**
   - **Text Vectorization:** We used a `TfidfVectorizer` to convert the bug descriptions (text data) into numerical features (vectors). This step transforms the textual data into a format that machine learning models can process.
   - **Categorical Encoding:** Categorical features, such as `Severity`, were one-hot encoded. This means that categorical variables were transformed into a binary matrix representation.
   - **Combining Features:** All numerical, categorical, and text-based features were combined into a single matrix (`X_combined`). This matrix is what the machine learning models will use to learn patterns in the data.
   - **Splitting Data:** The combined data was split into training and test sets. The training set is used to train the models, while the test set is used to evaluate their performance.

3. **Model Training and Prediction:**
   - **Model Selection:** We selected multiple models (Random Forest, Logistic Regression, Support Vector Machine, Naive Bayes) to train on the prepared data.
   - **Training:** Each model is trained using the training data (`X_train` and `y_train`). This step involves feeding the model data and allowing it to learn the patterns that distinguish between different labels (e.g., severity of bugs).
   - **Prediction:** After training, each model predicts labels on the test set (`X_test`). This is where the models use what they learned to make predictions on new data.
   - **Evaluation:** The predictions are compared against the actual labels (`y_test`) to evaluate how well the models performed. Metrics like accuracy, confusion matrix, and classification reports are used to measure performance.

4. **Visualization:**
   - **Confusion Matrix:** The confusion matrix for each model's predictions is plotted. This visualizes how many predictions were correct and where the model made mistakes, giving you a sense of each model's strengths and weaknesses.

### What We Are Doing:
- **Prediction Task:** The models are being trained to predict whether a software bug might be severe or not (or other categories defined in the labels). This is inferred from the combination of numerical, categorical, and text features related to each bug.
- **Why Multiple Models:** Different models have different strengths. By training multiple models, you can compare their performance and choose the best one for your specific task.
- **Visualization:** The visualization helps in understanding the prediction performance of each model. For example, a confusion matrix shows how often predictions were correct versus incorrect.

### Key Points:
- **Prediction of Software Bugs:** The core prediction task is to determine the severity (or any other outcome) of software bugs based on historical data. This allows developers to prioritize bugs that might be more critical.
- **Where Is the Prediction Happening?:** The prediction happens when the trained models are applied to the test data (`X_test`) to predict `y_test`. This gives us an idea of how the model would perform on new, unseen data.
