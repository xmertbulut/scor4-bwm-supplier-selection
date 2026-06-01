# scor4-bwm-supplier-selection
Graduation project implementing a digitalized supplier selection framework using SCOR 4.0, Best-Worst Method (BWM), and Gradient Boosting Regressor for automated procurement.

The primary objective of this project is to create a digital decision-support system that:
Uses SCOR 4.0 dimensions to structure procurement criteria.
Employs the Best-Worst Method (BWM) to calculate precise criteria weights with minimal expert cognitive load.
Utilizes Gradient Boosting Regressor (Machine Learning) to automate supplier ranking based on learned decision logic.

/data: Contains bwm_data.csv and supplier_data.csv (synthetic datasets).
/notebooks: Contains the primary Python script/pipeline.
/docs: Project documentation and final report materials.

Environment: Google Colab / Jupyter Notebook
Key Libraries:
Pandas: Data ingestion and preprocessing.
SciPy: Solving non-linear optimization for BWM.
Scikit-Learn: Gradient Boosting model training and evaluation.

How to Run??
Open the .py file in Google Colab.
Ensure the dataset files (.csv) are uploaded to the environment.
Run the cells sequentially to perform optimization and model training.
The system will output the final weight distribution and the MSE (Mean Squared Error) score of the model.
