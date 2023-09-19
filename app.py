import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
log_reg = pickle.load(open('log_reg.pkl','rb'))
sc = pickle.load(open('standSc.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

def calc_egfr_levels(egfr_val):
    if egfr_val>0 and egfr_val<15:
        egfr_lvl = 'Kidney Failure'
    elif egfr_val>=15 and egfr_val<=29:
        egfr_lvl = 'Severe decrease in GFR'
    elif egfr_val >= 30 and egfr_val <= 59:
        egfr_lvl = 'Moderate decrease in GFR'
    elif egfr_val >= 60 and egfr_val <= 89:
        egfr_lvl = 'Mild decrease in GFR'
    elif egfr_val >= 90:
        egfr_lvl = 'Kidney damage with normal or increased GFR'
    else:
        egfr_lvl = 'Error Calculating EGFR'
    return egfr_lvl
    


@app.route('/')
def home():
    return render_template('ditect.html')

@app.route('/predict',methods=['POST'])
def predict():
    tot_features = [x for x in request.form]
    tot_feat_num = [x for x in request.form.values()]
    feat_dict = {}
    for i,j in zip(tot_features, tot_feat_num):
        feat_dict[i] = j
    selected_feat =['age', 'blood_pressure', 'pus_cell_clumps', 'bacteria', 'serum_creatinine', 'haemoglobin', 'potassium', 'sodium', 'pus_cell', 'red_blood_cells', 'specific_gravity']
    needed_features = [feat_dict[x] for x in selected_feat]
    float_features = [float(x) for x in needed_features]
    final_features = [np.array(float_features)]
    pred = log_reg.predict( sc.transform(final_features) )
    if pred:
        if feat_dict['gender'] == 'F':
            gend_val = 0.742
        else:
            gend_val = 1
        egfr = np.round(175*(float(feat_dict['serum_creatinine'])**(-1.154))*(float(feat_dict['age'])**(-0.203))*gend_val, 2)
        lvl = calc_egfr_levels(egfr)
        return render_template('result.html', prediction=pred, eGFR = egfr, Lvl=lvl)
    else:
        return render_template('result.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)
