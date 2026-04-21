# 🔋 Battery Anomaly Detection System

An interactive web application built using Streamlit to detect abnormal battery usage patterns using Machine Learning.

## 🚀 Features

- Detects unusual battery drain behavior
- Uses Isolation Forest and Local Outlier Factor (LOF)
- Synthetic dataset generation + CSV upload support
- Real-time anomaly prediction
- Time-based analysis (hourly, daily trends)
- Interactive visualizations (PCA, heatmaps, histograms)

---

## 🧠 Machine Learning Models Used

1. Isolation Forest  
2. Local Outlier Factor (LOF)

---

## 📊 Input Features

- ScreenTime
- CPUUsage
- Apps
- BackgroundApps
- NetworkUsage
- BatteryDrop

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure
project1/
│── battery_anomaly_detection.py
│── README.md
│── requirements.txt

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
pip install -r requirements.txt

### Step 2: Run the app

---

## 📁 Dataset

- Default: Synthetic dataset (auto generated)
- Optional: Upload your own CSV with required columns

---

## 📸 Output

- Dashboard with anomaly detection summary
- Graphs and visual insights
- Real-time prediction panel

---

## 📌 Future Improvements

- Deploy on cloud (Streamlit Cloud / GCP)
- Add deep learning models
- Connect real mobile battery data

---

## 👩‍💻 Author

Supritha