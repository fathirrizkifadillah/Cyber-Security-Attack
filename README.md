# Cyber Security Attack Analysis & Classifier

An end-to-end data science and machine learning project focused on detecting and classifying cyber security attacks based on network traffic characteristics. This project includes a comprehensive Jupyter notebook for exploratory data analysis (EDA), model training (Decision Tree and Random Forest), and an interactive **Streamlit Dashboard** for real-time attack prediction and model evaluation.

---

## Project Overview

This project analyzes network traffic datasets to distinguish normal network activity from malicious activities, specifically identifying attack vectors such as **DDoS**, **BruteForce**, and other intrusions. By training machine learning classifiers, the system can predict the attack type in real-time based on connection parameters.

### Key Findings
* **The BruteForce Indicator**: Feature importance analysis shows that `failed_logins` is the ultimate indicator for **BruteForce** attacks. Classifying brute force attempts can be achieved reliably through simple thresholding of failed login attempts.
* **Model Generalization**: The **Random Forest Classifier** outperformed the single **Decision Tree**, demonstrating superior generalization and robustness against high-dimensional noise in network packet traffic. Both models achieved near-perfect F1-scores on classes with distinct signatures like DDoS and BruteForce.
* **Protocol Vulnerabilities**: Distinct patterns emerged comparing TCP and UDP protocols, with specific attack types heavily favoring one over the other due to connection-oriented vs. connectionless architecture signatures.

---

## Project Structure

```text
├── models/
│   ├── decision_tree_model.pkl      # Saved Decision Tree classifier
│   └── random_forest_model.pkl      # Saved Random Forest classifier
├── img/
│   └── Feature Importance Comparison.png  # Generated model feature importances
├── main.ipynb                       # Notebook for EDA, preprocessing, and model training
├── app.py                           # Multi-tab Streamlit dashboard application
├── Cyber_Security_Attack_Dataset_Exp.md # Detailed explanation of dataset features
├── requirements.txt                 # Project dependencies list
└── README.md                        # Project documentation (this file)
