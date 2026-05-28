FILES = {
    "2017": {
        "diabetes": "DIQ_J",
        "demo": "DEMO_J",
        "bmi": "BMX_J",
        "glucose": "GLU_J",
        "ghb": "GHB_J",
        "chol": "TCHOL_J",
        "hdl": "HDL_J",
        "bp": "BPX_J",
        "sleep": "SLQ_J",
        "occupation": "OCQ_J",
    },
    "2015": {
        "diabetes": "DIQ_I",
        "demo": "DEMO_I",
        "bmi": "BMX_I",
        "glucose": "GLU_I",
        "ghb": "GHB_I",
        "chol": "TCHOL_I",
        "hdl": "HDL_I",
        "bp": "BPX_I",
        "sleep": "SLQ_I",
        "occupation": "OCQ_I",
    },
    "2013": {
        "diabetes": "DIQ_H",
        "demo": "DEMO_H",
        "bmi": "BMX_H",
        "glucose": "GLU_H",
        "ghb": "GHB_H",
        "chol": "TCHOL_H",
        "hdl": "HDL_H",
        "bp": "BPX_H",
        "sleep": "SLQ_H",
        "occupation": "OCQ_H",
    },
}


BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.XPT"

KEEP: dict[str, str] = {
    "DIQ010": "Diabetes status",
    # istotne (bez LBXGLU)
    "LBXGH": "HbA1c - glycohemoglobin (%), long-term blood glucose indicator",
    "BMXBMI": "Body Mass Index (kg/m²)",
    "BMXWAIST": "Waist circumference (cm)",
    "RIDAGEYR": "Age at screening (years)",
    "LBXTC": "Total cholesterol (mg/dL)",
    "LBDHDD": "HDL cholesterol (mg/dL)",
    "BPXSY1": "Systolic blood pressure - 1st reading (mmHg)",
    "BPXDI1": "Diastolic blood pressure - 1st reading (mmHg)",
    "RIAGENDR": "Gender (1=Male, 2=Female)",
    # nieistotne
    "RIDEXMON": "Month of examination",
    "DMDEDUC2": "Education level (1-5 scale) - socioeconomic proxy",
    "DMDHHSIZ": "Total number of people in the household",
}
