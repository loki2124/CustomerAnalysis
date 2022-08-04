import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,ConfusionMatrixDisplay

import pickle
import joblib
import datetime
import xgboost
from collections import defaultdict
import datetime as dt


#uncomment (if commented) the following to load the artifacts for cloud before deployment 
# CHURN_MODEL  = pickle.load(open('/app/customeranalysis/models/churn_model.pickle', 'rb'))
# CHURN_SCALER = joblib.load('/app/customeranalysis/models/churn_scaler.save') 
# USAGE_MODEL = pickle.load(open('/app/customeranalysis/models/usage_model.pickle', 'rb'))
# BILLING_MODEL = pickle.load(open('/app/customeranalysis/models/billing_upgrade_model.pickle', 'rb'))
# SHAP_VALS_USAGE = Image.open('/app/customeranalysis/plots/SHAP_VALS_USAGE.jpeg')
# SHAP_VALS_BILLING = Image.open('/app/customeranalysis/plots/SHAP_VALS_BILLING.jpeg')
# CLV_VALS  = pd.read_csv('/app/customeranalysis/models/CLV.csv')

#uncomment (if commented) the following to load the artifacts for localhost before deployment 
CHURN_MODEL  = pickle.load(open('./models/churn_model.pickle', 'rb'))
CHURN_SCALER = joblib.load('./models/churn_scaler.save') 
USAGE_MODEL = pickle.load(open('./models/usage_model.pickle', 'rb'))
BILLING_MODEL = pickle.load(open('./models/billing_upgrade_model.pickle', 'rb'))
SHAP_VALS_USAGE = Image.open('./plots/SHAP_VALS_USAGE.jpeg')
SHAP_VALS_BILLING = Image.open('./plots/SHAP_VALS_BILLING.jpeg')
CLV_VALS  = pd.read_csv('./models/CLV.csv')

fig = plt.figure()

st.set_option('deprecation.showPyplotGlobalUse', False)

original_title = '<p style="font-family:Ariel; text-align:center; color:saddlebrown ; font-size:50px; background-color:#FEE1D1;opacity: 0.9;">Customer Analysis</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.text("")
st.text("")
st.text("")


def prepare_churn_data(df_cust_churn, scaler):

    """
        This function prepares data to be fed into the churn model. 
        It receives raw data and scaler artifact as input and returns cleaned data, 
        labels and their IDs.

    :param DataFrame df_cust_churn: Raw Dataset
    :param joblib scaler: Scaler to transform raw data to standardize
    :return: DataFrame X (clean data), Array y (label), Array CID (Customer ID)
    """
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
    X = scaler.transform(X)
    #get customer ID to tie back the prediction results
    CID = df_churn['CMU_ID_new']

    return X, y, CID


def prepare_usage_data(df):

    """
        This function prepares data to be fed into the usage model. 
        It receives raw data as input and returns cleaned data, 
        labels and their IDs.

    :param DataFrame df: Raw Dataset
    :return: DataFrame X (clean data), Array y (label), Array CID (Customer ID)
    """

    df3=df[(~df['deactivate_month'].isna()) & (~df['trial_start_month'].isna())]
    df3=df3.groupby('CMU_ID_new').agg({'deactivate_month':['max'],'trial_start_month':['min']})
    df3.columns=['max_deactivate_month','min_trial_start_month']
    
    df3=df3[(pd.to_datetime(df3['max_deactivate_month'])-pd.to_datetime(df3['min_trial_start_month'])).dt.days<60]
    df2=df[~df['CMU_ID_new'].isin(df3.index)]
        
    df2=df2[[
    'CMU_ID_new',   #Added by Loki to get CID
    'ram_capacity_gb_avg',
    'contract',
    'billing_model',
    'contract_tier',
    'continent',
    'market_segment',
    'support_cases_count'
    ]]
    df2=df2.dropna() 
    
    df2['target']=df2.apply(lambda y:1 if y['ram_capacity_gb_avg']>df2['ram_capacity_gb_avg'].median() else 0,axis=1 )

    df3 = pd.get_dummies(df2, columns = list(set(df2.columns)-{'target','ram_capacity_gb_avg','support_cases_count', 'CMU_ID_new'}))
    
    # load data
    fts=list(set(df3.columns)-{'target','ram_capacity_gb_avg', 'CMU_ID_new'}) 
    # split data into X and y
    X = df3[fts].values
    y = df3['target'].values
    
    CID = df2['CMU_ID_new']
    return X, y, CID


def get_most(x):
    hi, hiv, dc = None, 0, defaultdict(int)
    for _, v in x.iteritems():
        if v is not None and v != '':
            dc[v] += 1
            if dc[v] > hiv:
                hi, hiv = v, dc[v]
    return hi


def prepare_billing_data(df):

    """
        This function prepares data to be fed into the billing model. 
        It receives raw data as input and returns cleaned data, 
        labels and their IDs.

    :param DataFrame df: Raw Dataset
    :return: DataFrame X (clean data), Array y (label), Array CID (Customer ID)
    """

    #created a new column to record previou account tier
    df['prevAccType'] = df.sort_values(by=['account_month'], ascending=True)\
                           .groupby(['CMU_ID_new'])['account_contract_type'].shift(1)
    df['prevContract'] = df.sort_values(by=['account_month'], ascending=True)\
                           .groupby(['CMU_ID_new'])['contract_tier'].shift(1)
    
    if is_datetime(df['account_month']) == False:
        df['account_month'] = pd.to_datetime(df['account_month']).dt.tz_localize('UTC')
    df['create_month'] = pd.to_datetime(df['create_month'])
    df['diffMonths'] = (df['account_month'] - df['create_month']).dt.total_seconds()/2.628e6
    
    # whether those services are used all the time
    df['full_security'] = df.apply(lambda x: 1 if x.security_firstday and x.security_lastday else 0, axis=1)
    df['full_observability'] = df.apply(lambda x: 1 if x.observability_firstday and x.observability_lastday else 0, axis=1)
    df['full_search'] = df.apply(lambda x: 1 if x.enterpise_firstday and x.enterpise_lastday else 0, axis=1)
    
    # Mark records as billingUpgrade if they upgrade from monthly to annual/multi-year
    prev_acc_mask = df['prevAccType'] == 'monthly'
    curr_acc_mask = (df['account_contract_type'] == 'multi-year') | (df['account_contract_type'] == 'annual')
    billing_upgraders = df[prev_acc_mask & curr_acc_mask]['CMU_ID_new'].unique() #no of unique customers who upgraded : 1328/31061; 
    df['billingUpgrade'] = df['CMU_ID_new'].apply(lambda x: True if x in billing_upgraders else False)

    df['beforeBillingUpgrade'] = df.apply(lambda x: x.account_contract_type == 'monthly', axis=1)

    prev_std_mask = df['prevContract'] == 'standard'
    curr_plt_mask = (df['contract_tier'] == 'gold') | (df['contract_tier'] == 'platinum') | (df['contract_tier'] == 'enterprise')
    tier_upgraders = df[prev_std_mask & curr_plt_mask]['CMU_ID_new'].unique() #no of unique customers who upgraded : 432/31061;
    df['tierUpgrade'] = df['CMU_ID_new'].apply(lambda x: True if x in tier_upgraders else False)

    df['beforeTierUpgrade'] = df.apply(lambda x: x.contract_tier == 'standard', axis=1)

    # Delete all records other than monthly ones from dataset.
    df2 = df[df.beforeBillingUpgrade]
    df3 = df[df.beforeTierUpgrade]
    
    df_unique = df2.groupby(by = 'CMU_ID_new').agg({
        'arr' : 'mean',
        'ram_capacity_gb_avg' : 'mean',
        'disk_usage_gb_avg': 'mean',
        'support_cases_count': 'mean',
        'search_requests_24_avg': 'mean',
        'search_requests_24_lastday' : 'mean',
        'search_requests_24_firstday' : 'mean',
        'disk_usage_gb_firstday' : 'mean',
        'disk_usage_gb_lastday' : 'mean',
        'ram_capacity_gb_firstday' : 'mean',
        'ram_capacity_gb_lastday' : 'mean',
        'full_security' : 'mean',
        'full_observability' : 'mean',
        'full_search' : 'mean',
        'account_source' : get_most,
        'market_segment' : get_most,
    #    'continent' : get_most,
    })

    df_unique = df_unique.reset_index()
    df_unique['billingUpgrade'] = df_unique['CMU_ID_new'].apply(lambda x: True if x in billing_upgraders else False)
    
    df_unique = df_unique.dropna(axis=0, subset=[
        'arr',
        'ram_capacity_gb_avg',
        'disk_usage_gb_avg',
        'support_cases_count',
        'search_requests_24_avg',
        'search_requests_24_lastday',
        'search_requests_24_firstday',
        'disk_usage_gb_firstday',
        'disk_usage_gb_lastday',
        'ram_capacity_gb_firstday',
        'ram_capacity_gb_lastday',
        'account_source',
        'market_segment'
    ])

    # billingUpgrade are those accounts who have upgraded from monthly to annual
    df_unique['billingUpgrade'] = df_unique['CMU_ID_new'].apply(lambda x: True if x in billing_upgraders else False)
#     del df_unique['CMU_ID_new']  
    
    df_unique.dropna(inplace=True)

    # One-Hot encode the categorical columns
    df_unique = pd.get_dummies(df_unique, columns = ['account_source', 'market_segment'])

    y = np.array(df_unique['billingUpgrade'])
    fts = list(set(df_unique.columns)-{'billingUpgrade', 'CMU_ID_new'})
#     X = df_unique.loc[:, df_unique.columns != 'billingUpgrade']
    X = df_unique[fts]
    CID = df_unique['CMU_ID_new']
    
    return X, y, CID


def model_prediction(X,y = None, model = 'CHURN', thres = 0.5):

    """
        This function provides model prediction on the dataset provided and
        also returns the predictions, confusion matrix, precision, recall and F1-score

    :param DataFrame X: Clean Data
    :param Array y: Class Label
    :param str model: Model Type
    :param str thres: Threshold (Propensity Rate) 
    :return: Array y_pred: Model Prediction
    :return Float precision: Precision
    :return Float recall: Recall
    :return Float fscore: F1-Score
    :return Float accuracy: Accuracy
    :return Array cf_matrix: Confusion Matrix
    """

    y_pred = None
    if model == 'CHURN':
        # predict probabilities
        y_pred = (CHURN_MODEL.predict_proba(X)[:,1] >= thres).astype(bool)
    elif model == 'USAGE':
        y_pred = (USAGE_MODEL.predict(X))
    
    elif model == 'BILLING':
        y_pred = (BILLING_MODEL.predict(X))

    if y_pred is not None:
        cf_matrix = confusion_matrix(y, y_pred)
        print(cf_matrix)
        recall = np.round(cf_matrix[1][1]/(cf_matrix[1][1] + cf_matrix[0][1]),2)
        precision = np.round(cf_matrix[1][1]/(cf_matrix[1][1] + cf_matrix[1][0]),2)
        fscore = np.round((2 * precision * recall) / (precision + recall),2)
        print(recall, precision, fscore)

        accuracy = np.round(accuracy_score(y, y_pred)*100, 2)
        print(accuracy)
    
    return y_pred, precision, recall, fscore, accuracy, cf_matrix

def plot_metrics(cf_matrix, class_names = None):

        """
            This function will plot the confusion matrix from the model prediction on the UI
        :param Array cf_matrix: Confusion Matrix
        :param List class_names: Class Labels to be printed
        """

        st.subheader("Confusion Matrix") 
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                              display_labels= class_names)
        disp = disp.plot()
        st.pyplot()
    

@st.cache
def convert_df(df):
    """
        This function will convert the dataframe to csv before downloading for the user
    :param DataFrame df: Prediction DataFrame
    :return Dataframe: Dataframe converted to csv formt 
    """
    return df.to_csv().encode('utf-8')


def main():
   
    selectbox_text = '<p style="font-family:Ariel; text-align:left; color:peru; font-size:30px;">Types of Model</p>'
    st.sidebar.markdown(selectbox_text, unsafe_allow_html=True)
    option = st.sidebar.selectbox("", ('Churn', 'Usage', 'Upgrade'))
    st.sidebar.write('You selected:', option)
    st.text("")
    st.text("")
    st.text("")

    #file uploader 
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(df.columns[0], axis=1)
        st.write(df)

    #check for the option selected by the user 
    if option == 'Churn':
            propensity_rate = st.slider('Propensity Rate', 0.00, 1.00, 0.50)
            col1, col2, col3 , col4, col5 = st.columns(5)
            with col3:
                try:
                    pred_button = st.button('Predict Churn')
                except:
                    print('Please upload the file first...')
                    st.text("Please upload the file first...")
            if pred_button:
                with st.spinner('Churn Model Working....'):
                    X, y, cid = prepare_churn_data(df, CHURN_SCALER)
                    y_pred, precision, recall, fscore, accuracy, cf = model_prediction(X, y, thres = propensity_rate, model = 'CHURN')
                    agg_results = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'F-Score': [fscore], 'Accuracy': [accuracy]}, index= ['Model Scores'])
                    customer_predictions = pd.DataFrame({'CMU_ID_new': cid, 'Churn_Prediction': y_pred})
                    customer_predictions_merged = pd.merge(customer_predictions, CLV_VALS, on = ['CMU_ID_new'])
                    customer_predictions_csv = convert_df(customer_predictions_merged)
                    st.subheader('Model Results')
                    st.table(agg_results)
                    plot_metrics(cf, class_names = ['No Churn', 'Churn'])
                    st.download_button('Download Churn Predictions', customer_predictions_csv, 'churn_prediction.csv', 'text/csv')
    if option == 'Usage':
            col1, col2, col3 , col4, col5 = st.columns(5)
            with col3:
                pred_button = st.button('Predict Usage')
            if pred_button:
                with st.spinner('Usage Model Working....'):
                    X, y, cid = prepare_usage_data(df)
                    y_pred, precision, recall, fscore, accuracy, cf = model_prediction(X, y, model = 'USAGE', thres = 0.5)
                    agg_results = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'F-Score': [fscore], 'Accuracy': [accuracy]}, index= ['Model Scores'])
                    customer_predictions = pd.DataFrame({'CMU_ID_new': cid, 'Usage_Prediction': y_pred})
                    customer_predictions_csv = convert_df(customer_predictions)
                    st.subheader('Model Results')
                    st.table(agg_results)
                    plot_metrics(cf, class_names = ['No Use', 'Use'])
                    st.text("")
                    st.subheader("Feature Importance") 
                    st.image(SHAP_VALS_USAGE, caption='Important Features for Billing')
                    st.download_button('Download Usage Predictions', customer_predictions_csv, 'usage_prediction.csv', 'text/csv')

    if option == 'Upgrade':
            col1, col2, col3 , col4, col5 = st.columns(5)
            with col3:
                pred_button = st.button('Predict Billing Upgrade')
            if pred_button:
                with st.spinner('Billing Upgrade Model Working....'):
                    X, y, cid = prepare_billing_data(df)
                    y_pred, precision, recall, fscore, accuracy, cf = model_prediction(X, y, model = 'BILLING', thres = 0.5)
                    agg_results = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'F-Score': [fscore], 'Accuracy': [accuracy]}, index= ['Model Scores'])
                    customer_predictions = pd.DataFrame({'CMU_ID_new': cid, 'Billing_Prediction': y_pred})
                    customer_predictions_csv = convert_df(customer_predictions)
                    st.subheader('Model Results')
                    st.table(agg_results)
                    plot_metrics(cf, class_names = ['No Upgrade', 'Upgrade'])
                    st.text("")
                    st.subheader("Feature Importance") 
                    st.image(SHAP_VALS_BILLING, caption='Important Features for Usage')
                    st.download_button('Download Upgrade Predictions', customer_predictions_csv, 'upgrade_prediction.csv', 'text/csv')



if __name__ == "__main__":
    main()
