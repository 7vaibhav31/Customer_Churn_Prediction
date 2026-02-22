# Customer Churn Prediction

This repository contains code for a customer churn prediction project.

Structure:
- `src/` - preprocessing and training scripts
  - `preprocessing.py` - data loading and preprocessing
  - `train_model.py` - model definition and training
- `app/` - Streamlit application
  - `streamlit_app.py` - web UI for single and batch predictions
- `test_data/` - sample CSV files for batch testing
- `requirements.txt` - project dependencies

Usage
1. Create and activate a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model (optional):

```powershell
python src/train_model.py
```

4. Run the Streamlit app:

```powershell
python -m streamlit run app/streamlit_app.py
```

Sample batch file: `test_data/sample_batch.csv`
