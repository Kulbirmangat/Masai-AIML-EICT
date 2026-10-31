# Section 1: How to run
To run this code, you must - 
* **Clone the repository and enter the project directory.**
  ```bash
  git clone <https://github.com/Kulbirmangat/Masai-AIML-EICT>
  cd Masai-AIML-EICT
  ```
* **Install all pinned dependencies.**
  ```bash
  pip install -r requirements.txt
  ```
* **Run the files in order to generate the data/, models/, and outputs/ directory.**
  ```bash
  python generate_data.py
  python src/shipment_prep.py
  python part_a_clean_eda.py
  python part_b_regression.py
  python part_c_classification.py
  python part_d_segments.py
  python package_model.py
  python predict_delay.py
  ```
### Environment & Model Serialization Guardrails
* **Scikit-Learn Version Dependency**: The trained model artifact `models/delay_model.pkl` is serialized via `joblib` and relies strictly on the `scikit-learn` version specified in `requirements.txt` (1.9.1).
* **Grader Fallback**: If the evaluation environment runs a different version of `scikit-learn`, do not load the pre-committed pickle file directly. Re-running `python package_model.py` inside that environment will re-serialize the pipeline locally, ensuring seamless compatibility and preventing unpickling errors.
# Section 2: Data Findings
* **Service Type Delay Leader**
