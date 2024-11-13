# Sentiment Analysis of Twitter Data

This project demonstrates a sentiment analysis pipeline applied to Twitter data using Python. The analysis involves data preprocessing, model selection, training, and evaluation, utilizing both traditional machine learning models and deep learning techniques. 

## Project Structure

The notebook covers the following sections:
1. **Data Preprocessing**: Loading and cleaning Twitter data, tokenizing text, and preparing the data for model training.
2. **Exploratory Data Analysis (EDA)**: Visualizing the distribution of sentiment classes and key insights about the dataset.
3. **Feature Engineering**: Using TF-IDF vectorization for traditional models and word embeddings for deep learning models.
4. **Model Training and Evaluation**:
    - Traditional Machine Learning Models: Logistic Regression, Naive Bayes, Support Vector Machines (SVM).
    - Deep Learning Models: LSTM networks implemented with TensorFlow and Keras.
5. **Results and Analysis**: Comparing model performance using accuracy, precision, recall, F1-score, and visualizing confusion matrices.

## Dependencies

To run the notebook, you'll need the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `tensorflow`

You can install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## How to Run the Notebook

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory and open the notebook:
   ```bash
   jupyter notebook TAD_coursework_2773581G.ipynb
   ```
3. Follow the notebook cells sequentially to perform the analysis.

## Results

The results indicate that both traditional ML models and the LSTM network achieve good performance on sentiment prediction. Comparative analysis of the models is provided, including insights into accuracy and classification metrics.

## Future Work

Potential future improvements include:
- Expanding the dataset with more diverse sentiment labels.
- Experimenting with other neural network architectures, such as GRU or transformer models.
- Fine-tuning hyperparameters for optimal model performance.
