## How to Run the Project ▶️
To execute the project, follow these steps:

1. **Create and activate a virtual environment:**  
   - Navigate to the project folder (if not already there):
     ```sh
     cd final_project/
     ```
   - Create a virtual environment:
     ```sh
     python -m venv myenv
     ```
   - Activate the virtual environment:
     - On macOS/Linux:
       ```sh
       source myenv/bin/activate
       ```
     - On Windows:
       ```sh
       myenv\Scripts\activate
       ```

2. **Install dependencies:**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the application:**  
   ```sh
   streamlit run app.py
   ```

## Project Context 📚
This project is part of an investigation into **“Intelligent Decision-Making with Machine Learning using Explainability Techniques for Time Series Problems.”** The work was supervised by Professor Fábio Silva and aimed at researching machine learning models for anomaly detection in time series data.

### Objectives 🎯
The primary goal of this project is to enhance the explainability of machine learning models applied to anomaly detection in time series. Specific objectives include:
- Studying time series data.
- Exploring Explainable AI (XAI) frameworks.
- Analyzing black-box and white-box models.
- Developing proof-of-concept implementations.
- Conducting a case study using 5G network datasets.

### Results 📊
The project involved training models such as **Naïve Bayes** and **Decision Tree** on pre-processed datasets. Explainability techniques like **SHAP (Shapley Additive Explanations)** and **LIME (Local Interpretable Model-Agnostic Explanations)** were applied to interpret the decision-making process of these models.

Additionally, a **web platform** was developed using **Streamlit**, allowing users to:
- Import datasets and visualize time-series data through interactive dashboards.
- Analyze trained models and their explainability outputs.
- Generate predictions with trained models and view **LIME-based explanations** for each prediction.
