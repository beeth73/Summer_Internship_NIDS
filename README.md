# Network Intrusion Detection System (NIDS) using Machine Learning

![Cybersecurity](https://img.shields.io/badge/domain-cybersecurity-blue)
![Python](https://img.shields.io/badge/python-3.11-blueviolet)
![Scikit-learn](https://img.shields.io/badge/sklearn-passing-green)
![Pandas](https://img.shields.io/badge/pandas-passing-brightgreen)
![XGBoost](https://img.shields.io/badge/xgboost-passing-purple)

This repository contains the complete project for an intelligent **Network Intrusion Detection System (NIDS)** developed as part of a summer internship at **Oil and Natural Gas Corporation Limited (ONGC), Mumbai**. The project leverages a comprehensive machine learning pipeline to classify network traffic as either "Benign" or "Attack" using the UNSW-NB15 dataset.

---

## 📄 Table of Contents

- [Project Overview](#-project-overview)
- [Data Source](#-data-source)
- [Problem Statement](#-problem-statement)
- [Key Features & Methodology](#-key-features--methodology)
- [Final Results & Conclusion](#-final-results--conclusion)
- [Repository Contents](#-repository-contents)
- [Technologies & Libraries Used](#-technologies--libraries-used)
- [How to Run the Notebook](#-how-to-run-the-notebook)
- [Acknowledgement](#-acknowledgement)

---

## 📖 Project Overview

In today's digital landscape, traditional signature-based intrusion detection systems often fail to detect novel and polymorphic (zero-day) attacks. This project addresses this critical security gap by building a data-driven NIDS that learns from network patterns to identify threats. The entire workflow, from data ingestion to model evaluation, is documented in the `hope.ipynb` notebook.

## 📊 Data Source

The dataset used for this project is the **UNSW-NB15 dataset**, created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). It is widely used as a benchmark for evaluating NIDS performance.

The raw CSV files (`UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv`) and the `NUSW-NB15_features.csv` file were downloaded from the official source provided by UNSW Sydney:

- **[UNSW-NB15 Dataset Download Link](https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset%2FCSV%20Files&ga=1)**

---

## 🎯 Problem Statement

The core challenge was to design and implement a machine learning pipeline capable of accurately classifying network traffic on the complex, high-dimensional, and severely imbalanced **UNSW-NB15 dataset**. The system needed to effectively distinguish between normal traffic and a wide range of modern cyberattacks, overcoming the limitations of static, rule-based systems.

## 🛠️ Key Features & Methodology

The project followed a structured, end-to-end data science lifecycle designed for efficiency and performance on consumer-grade hardware.

1.  **Data Preprocessing:**
    *   Merged four raw CSV files into a single DataFrame of over 2.5 million records.
    *   Cleaned the data by dropping irrelevant columns, removing over **490,000 duplicate records**, and correcting data types.
    *   Applied **log transformation** to normalize skewed numerical features.

2.  **Feature Engineering & Selection:**
    *   **Limited the cardinality** of high-variance categorical features (like `service`) to prevent memory errors during encoding.
    *   Trained a **RandomForest** model on a stratified sample of the data to calculate feature importances.
    *   Selected the **top 66 most predictive features** to reduce dimensionality and improve model efficiency.

3.  **Class Imbalance Handling:**
    *   Systematically compared two key strategies:
        1.  Training models on the imbalanced data using model-native parameters (`scale_pos_weight`).
        2.  Training models on a perfectly balanced dataset created using **SMOTE** (Synthetic Minority Over-sampling Technique).

4.  **Comparative Model Evaluation:**
    *   Trained and benchmarked four distinct classifiers: **Logistic Regression, RandomForest, XGBoost, and LightGBM**.
    *   Evaluated all 8 scenarios (4 models x 2 strategies) on a held-out test set using metrics like Precision, Recall, F1-Score, and ROC AUC.

---

## 🏆 Final Results & Conclusion

The comprehensive analysis revealed clear trade-offs between detection rates, false alarms, and training speed.

-   **Best Detection Rate (Lowest False Negatives):** The **LightGBM (No SMOTE)** model proved to be the most vigilant, missing only 26 out of ~64,000 attacks in the test set.
-   **Best Balance (Highest F1-Score):** The **RandomForest (No SMOTE)** model achieved the best balance of Precision and Recall.
-   **Recommended Overall Model:** The **RandomForest (With SMOTE)** model was identified as the best overall solution from the analysis, offering an excellent combination of high detection rates and reliability.

This project underscores the power of tree-based ensembles for intrusion detection and highlights the critical importance of selecting a model based on specific security priorities.

---

## 📁 Repository Contents

-   `hope.ipynb`: The main Jupyter Notebook containing the complete, optimized Python code for the entire data pipeline.
-   `Bhushan_NIDS_ONGC_SDC_organized.pdf`: The detailed project report submitted for the internship.
-   `Certificate of Participation.pdf`: The official certificate of participation for the summer internship at ONGC.

---

## 💻 Technologies & Libraries Used

-   **Language:** Python 3.11
-   **Core Libraries:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn, XGBoost, LightGBM
-   **Imbalance Handling:** Imbalanced-learn (for SMOTE)
-   **Visualization:** Matplotlib, Seaborn
-   **Environment:** Jupyter Lab (via Anaconda)

---

## 🚀 How to Run the Notebook

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/beeth73/Summer_Internship_NIDS.git
    cd Summer_Internship_NIDS
    ```

2.  **Set up the environment:** It is recommended to use `conda` to manage dependencies.
    ```bash
    # Create and activate a new conda environment
    conda create --name nids_env python=3.11
    conda activate nids_env

    # Install necessary packages
    pip install pandas numpy matplotlib seaborn jupyterlab scikit-learn imbalanced-learn xgboost lightgbm
    ```

3.  **Download the Dataset:**
    Download the required CSV files from the official **[UNSW-NB15 Dataset Source](https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset%2FCSV%20Files&ga=1)**. Place the following files in the root directory of this project:
    - `UNSW-NB15_1.csv`
    - `UNSW-NB15_2.csv`
    - `UNSW-NB15_3.csv`
    - `UNSW-NB15_4.csv`
    - `NUSW-NB15_features.csv`

4.  **Launch Jupyter Lab:**
    ```bash
    jupyter lab
    ```

5.  Open `hope.ipynb` and run the cells sequentially from top to bottom.

---

## 🙏 Acknowledgement

This project was completed under the invaluable guidance of my mentor, **Mr. Arvind Kumar (Manager, Programming)**, at the Database Group, WOB, ONGC, Mumbai. I am immensely grateful for the opportunity and the support provided by ONGC.
