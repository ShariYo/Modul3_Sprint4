from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI


# Expected Imput
class LoanApplication(BaseModel):
    sk_id_curr: int
    name_contract_type: object
    code_gender: object
    flag_own_car: object
    flag_own_realty: object
    cnt_children: int
    amt_annuity: float
    name_type_suite: object
    name_income_type: object
    name_education_type: object
    name_family_status: object
    name_housing_type: object
    region_population_relative: float
    client_age: float
    years_employed: float
    years_registration: float
    years_id_publish: float
    flag_mobil: object
    flag_emp_phone: object
    flag_work_phone: object
    flag_cont_mobile: object
    flag_phone: object
    flag_email: object
    occupation_type: object
    cnt_fam_members: int
    region_rating_client_w_city: int
    weekday_appr_process_start: object
    hour_appr_process_start: int
    reg_region_not_live_region: object
    reg_region_not_work_region: object
    live_region_not_work_region: object
    reg_city_not_live_city: object
    reg_city_not_work_city: object
    live_city_not_work_city: int
    organization_type: object
    ext_source_2: float
    ext_source_3: float
    def_30_cnt_social_circle: int
    obs_60_cnt_social_circle: int
    def_60_cnt_social_circle: int
    days_last_phone_change: float
    flag_document_2: object
    flag_document_3: object
    flag_document_4: object
    flag_document_5: object
    flag_document_6: object
    flag_document_7: object
    flag_document_8: object
    flag_document_9: object
    flag_document_10: object
    flag_document_11: object
    flag_document_12: object
    flag_document_13: object
    flag_document_14: object
    flag_document_15: object
    flag_document_16: object
    flag_document_17: object
    flag_document_18: object
    flag_document_19: object
    flag_document_20: object
    flag_document_21: object
    amt_req_credit_bureau_hour: int
    amt_req_credit_bureau_day: int
    amt_req_credit_bureau_week: int
    amt_req_credit_bureau_mon: int
    amt_req_credit_bureau_qrt: int
    amt_req_credit_bureau_year: int
    active_credit_count: float
    total_debt_all: object
    prol_credits: object
    credit_to_income: float


# Expected output
class PredictionOut(BaseModel):
    default_proba: float


# ----------------------------------------------------------------#


# Load model (mine is different)
model = joblib.load(r"D:\IT_projects\Turing_Colledge\Modul3\Sprint4\project\model.pkl")

# Start the app
app = FastAPI()


# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: LoanApplication):
    cust_df = pd.DataFrame([payload.dict()])
    preds = model.predict_proba(cust_df)[0, 1]
    result = {"default_proba": preds}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
