import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

import pickle
import joblib
import datetime


#loading the model
CHURN_MODEL  = pickle.load(open('../models/churn_model.pickle', 'rb'))
CHURN_SCALER = joblib.load('../models/churn_scaler.save') 

fig = plt.figure()


original_title = '<p style="font-family:Ariel; text-align:center; color:saddlebrown ; font-size:50px; background-color:#FEE1D1;opacity: 0.9;">Customer Analysis</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.text("")
st.text("")
st.text("")



def prepare_data(df_cust_churn):
    #converting datetime columns using pandas datetime 
    df_cust_churn['month_of_year'] = pd.to_datetime(df_cust_churn['month_of_year'])
    df_cust_churn['account_month'] = pd.to_datetime(df_cust_churn['account_month'])
    df_cust_churn['deactivate_month'] = pd.to_datetime(df_cust_churn['deactivate_month'])
    df_cust_churn['first_ingestion_month'] = pd.to_datetime(df_cust_churn['first_ingestion_month'])
    df_cust_churn['create_month'] = pd.to_datetime(df_cust_churn['create_month'])
    df_cust_churn['conversion_month'] = pd.to_datetime(df_cust_churn['conversion_month'])
    df_cust_churn['trial_end_month'] = pd.to_datetime(df_cust_churn['trial_end_month'])
    df_cust_churn['trial_start_month'] = pd.to_datetime(df_cust_churn['trial_start_month'])

    #aggregate customer level data using groupby
    df_cust = df_cust_churn.groupby(['CMU_ID_new','account_source','contract','contract_tier','continent','conversion_month'], as_index = False).sum()
    df_cust = df_cust.drop(['previous_month_arr','previous_days_in_month'],axis=1)
    df_duration = df_cust_churn.copy()
    df_duration = df_duration[df_duration['churned_in_month_bool']==1]
    df_duration['customer_duration'] = round(((df_duration['deactivate_month'] - df_duration['conversion_month'])/np.timedelta64(1, 'M')))
    columns_to_drop = [ 'month_of_year', 'arr',
        'previous_month_arr', 'days_in_month', 'previous_days_in_month',
        'arr_change_type', 'billing_model',
        'lane_change_bool', 'lane_change_detail', 'billing_model_change_bool',
        'tier_change_bool', 'churned_in_month_bool', 'account_month',
        'deactivate_month', 'first_ingestion_month', 'account_contract_type',
        'account_tier', 'create_month',
        'trial_end_month', 'trial_start_month', 'market_segment',
        'support_cases_count', 'account_sfdc_cases_count_s1',
        'account_sfdc_cases_count_s2', 'account_sfdc_cases_count_s3',
        'account_sfdc_industry', 'account_sfdc_sector',
        'account_sfdc_mkt_registration_source', 'account_sfdc_use_case',
        'deployment_month', 'disk_usage_gb_firstday',
        'ram_capacity_gb_firstday', 'search_requests_24_firstday',
        'enterpise_firstday', 'observability_firstday', 'security_firstday',
        'disk_usage_gb_lastday', 'ram_capacity_gb_lastday',
        'search_requests_24_lastday', 'enterpise_lastday',
        'observability_lastday', 'security_lastday', 'disk_usage_gb_avg',
        'ram_capacity_gb_avg', 'search_requests_24_avg', 'Paying_Customer_term']
    df_duration = df_duration.drop(columns_to_drop,axis=1)

    #joining two tables based on selected columns
    df_churn = pd.merge(df_cust,df_duration, on=['CMU_ID_new','account_source','contract','contract_tier','continent','conversion_month'], how='left')

    selection_condition = pd.isna(df_churn["customer_duration"])
    df_churn["customer_duration"].loc[selection_condition] = (df_churn["conversion_month"].loc[selection_condition].apply(lambda x: (datetime.datetime(2022,4,1) - datetime.datetime(x.year,x.month,x.day))))//np.timedelta64(1, 'M')
    df_churn = df_churn[df_churn['customer_duration']>2]

    X_continuous = df_churn[df_churn.columns.difference(['churned_in_month_bool','CMU_ID_new','conversion_month','account_source','contract','contract_tier','continent'])]
    X_discrete = pd.get_dummies(df_churn.loc[:,['account_source','contract','contract_tier','continent']], prefix_sep = "::", drop_first = True)
    X = X_continuous.join(X_discrete)

    y = df_churn.loc[:,df_churn.columns=="churned_in_month_bool"]

    #MinMax Standardization
    X = CHURN_SCALER.transform(X)
    return X, y



def churn_prediction(X,y = None, thres = 0.5):
    # predict probabilities
    y_pred = (CHURN_MODEL.predict_proba(X)[:,1] >= thres).astype(bool)
    cf_matrix = confusion_matrix(y, y_pred)
    print(cf_matrix)

    recall = np.round(cf_matrix[1][1]/(cf_matrix[1][1] + cf_matrix[0][1]),2)
    precision = np.round(cf_matrix[1][1]/(cf_matrix[1][1] + cf_matrix[1][0]),2)
    fscore = np.round((2 * precision * recall) / (precision + recall),2)
    print(recall, precision, fscore)

    accuracy = np.round(accuracy_score(y, y_pred)*100, 2)
    print(accuracy)

    return precision, recall, fscore, accuracy


def main():
   
    selectbox_text = '<p style="font-family:Ariel; text-align:left; color:peru; font-size:30px;">Types of Model</p>'
    st.sidebar.markdown(selectbox_text, unsafe_allow_html=True)
    option = st.sidebar.selectbox("", ('Churn', 'Usage', 'Upgrade'))
    st.sidebar.write('You selected:', option)
    st.text("")
    st.text("")
    st.text("")

    if option == 'Churn':
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.drop(df.columns[0], axis=1)
            st.write(df)

            propensity_rate = st.slider('Propensity Rate', 0.00, 1.00, 0.05)
            col1, col2, col3 , col4, col5 = st.columns(5)
            with col3:
                pred_button = st.button('Predict Churn')
            if pred_button:
                with st.spinner('Churn Model Working....'):
                    X, y = prepare_data(df)
                    precision, recall, fscore, accuracy = churn_prediction(X, y, propensity_rate)
                    results = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'F-Score': [fscore], 'Accuracy': [accuracy]}, index= ['Model Scores'])
                    st.table(results)



if __name__ == "__main__":
    main()
