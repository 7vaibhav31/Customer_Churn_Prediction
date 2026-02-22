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

3. Train the model (recommended run as module to use package imports):

```powershell
python -m src.train_model
```

4. Run the Streamlit app (start with the venv Python):

```powershell
.\venv\Scripts\python -m streamlit run app/streamlit_app.py
```

Sample batch file: test_data/sample_batch.csv

Push to GitHub

- Create a remote repository on GitHub and add it as a remote:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

- If you prefer using a Personal Access Token (PAT), use the URL format:

```
https://<TOKEN>@github.com/<your-username>/<repo-name>.git
```

CI

- This repo includes a simple GitHub Actions workflow `.github/workflows/python-lint.yml` that runs `flake8` on `src/` on push and pull requests.

Notes

- Run Streamlit with the venv Python to ensure TensorFlow is available.
- If you want, I can add a full CI that installs dependencies and runs tests — tell me if you'd like that.

Deploying to Streamlit Community Cloud

- Go to https://share.streamlit.io and sign in.
- Click "New app" → Connect to GitHub → select the `7vaibhav31/Customer_Churn_Prediction` repository.
- Set the app file path to `app/streamlit_app.py` and the branch to `main`.
- Use the default Python environment, or set `Python` to `3.10` in the UI if available.

Notes about model artifacts

- This repo includes the trained artifacts `preprocessor.joblib`, `pipeline.joblib`, and `keras_model.keras` in the repository root so the Streamlit app can load them at runtime. If you retrain the model locally, re-run `src/train_model.py` and commit the updated artifacts.

