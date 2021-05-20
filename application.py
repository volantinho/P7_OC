# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:58:41 2021

@author: VOLANTE
"""

import pandas as pd
import streamlit as st
import numpy as np
import pickle
import os

# Files importations####################################################################################################

# 12 statistics table (12 clusters)

stat0 = pd.read_csv(os.path.join('.', 'stat0.csv'), index_col=0)
stat1 = pd.read_csv(os.path.join('.', 'stat1.csv'), index_col=0)
stat2 = pd.read_csv(os.path.join('.', 'stat2.csv'), index_col=0)
stat3 = pd.read_csv(os.path.join('.', 'stat3.csv'), index_col=0)
stat4 = pd.read_csv(os.path.join('.', 'stat4.csv'), index_col=0)
stat5 = pd.read_csv(os.path.join('.', 'stat5.csv'), index_col=0)
stat6 = pd.read_csv(os.path.join('.', 'stat6.csv'), index_col=0)
stat7 = pd.read_csv(os.path.join('.', 'stat7.csv'), index_col=0)




# load the model from disk
km = pickle.load (open ('clustering_8', 'rb'))
XGB = pickle.load (open ('XGBClassifier', 'rb'))
    

# Importing our global stats
stat = pd.read_csv(os.path.join('.', 'global_stats.csv'), index_col=0)

# Importing our X_train and y_train

y_train = pd.read_csv(os.path.join('.', 'y_train.csv'), index_col=0)
X_train = pd.read_csv(os.path.join('.', 'X_train.csv'), index_col=0)
############################################################################################################################

st.title('Prêt à dépenser BANK')

# title and sub-title
st.write('''
# Will your LOAN be approved?

''')

# Title on sidebar
st.sidebar.header('Your informations')
st.sidebar.write('Please respect the order!')



def user_input():
    '''Creating a function which generates a DataFrame, accordind to consumer's inputs '''
    
    AGE = st.sidebar.slider('Your age', 21, 69, 40)
    AMT_INCOME_TOTAL = st.sidebar.number_input( 'Annual Income (Between {}$ and {}$)'.format(stat['AMT_INCOME_TOTAL']['min'],  stat['AMT_INCOME_TOTAL']['max']) , min_value = stat['AMT_INCOME_TOTAL']['min'] , max_value = stat['AMT_INCOME_TOTAL']['max'])
    DAYS_EMPLOYED = st.sidebar.number_input('Days employed (Between {} and {})'.format(int(stat['DAYS_EMPLOYED']['min']), int(stat['DAYS_EMPLOYED']['max'])),  min_value = int(stat['DAYS_EMPLOYED']['min']),  max_value = int(stat['DAYS_EMPLOYED']['max']))
    DAYS_REGISTRATION = st.sidebar.number_input('Last registration change (in days, between {} and {})'.format( int(stat['DAYS_REGISTRATION']['min']), int(stat['DAYS_REGISTRATION']['max'])), min_value = int(stat['DAYS_REGISTRATION']['min']), max_value = int(stat['DAYS_REGISTRATION']['max']))
    DAYS_ID_PUBLISH = st.sidebar.number_input('Last Identity document change (in days, between {} and {})'.format(int(stat['DAYS_ID_PUBLISH']['min']), int(stat['DAYS_ID_PUBLISH']['max'])), min_value = int(stat['DAYS_ID_PUBLISH']['min']), max_value = int(stat['DAYS_ID_PUBLISH']['max']))
    AMT_GOODS_PRICE = st.sidebar.number_input('Good price Amount (Between {}$ and {}$ allowed)'.format(stat['AMT_GOODS_PRICE']['min'], stat['AMT_GOODS_PRICE']['max']), min_value = stat['AMT_GOODS_PRICE']['min'], max_value = stat['AMT_GOODS_PRICE']['max'])
    AMT_CREDIT = st.sidebar.number_input('Credit Amount', min_value = stat['AMT_CREDIT']['min'], max_value = stat['AMT_CREDIT']['max'])
    AMT_ANNUITY = st.sidebar.number_input('Annuity Amount', min_value = stat['AMT_ANNUITY']['min'], max_value = stat['AMT_ANNUITY']['max'])
    EXT_SOURCE_2 = st.sidebar.slider('Extern source 2', stat['EXT_SOURCE_2']['min'], stat['EXT_SOURCE_2']['max'], 0.53)
    EXT_SOURCE_3 = st.sidebar.slider('Extern source 3', stat['EXT_SOURCE_3']['min'], stat['EXT_SOURCE_3']['max'], 0.51)
    
    # Creating selectbox for rating region 
    values1 = ['<select>',1, 2, 3]
    default_ix_1 = values1.index(1)
    REGION_RATING_CLIENT_W_CITY = st.sidebar.selectbox('Rating your region (taking city into account)', values1, index=default_ix_1)
    
    # Creating selectbox for REG_CITY_NOT_LIVE_CITY
    values2 = ['<select>',0, 1]
    default_ix_2 = values2.index(0)
    REG_CITY_NOT_LIVE_CITY = st.sidebar.selectbox('If your permanent adress does not match your contact adress select "1". Else select "0"', values2, index=default_ix_2)
    
     # Creating selectbox for REG_CITY_NOT_WORK_CITY
    values2 = ['<select>',0, 1]
    default_ix_2 = values2.index(0)
    REG_CITY_NOT_WORK_CITY = st.sidebar.selectbox('If your permanent adress does not match your work adress select "1". Else select "0"', values2, index=default_ix_2)
    
    
     # Creating selectbox for LIVE_CITY_NOT_WORK_CITY
    values2 = ['<select>',0, 1]
    default_ix_2 = values2.index(0)
    LIVE_CITY_NOT_WORK_CITY = st.sidebar.selectbox('If your contact adress does not match your work adress select "1". Else select "0"', values2, index=default_ix_2)
    
    
    
    # Creating a dictionary
    data = {'AGE' : AGE,
            'AMT_INCOME_TOTAL' : AMT_INCOME_TOTAL,
            'DAYS_EMPLOYED' : DAYS_EMPLOYED,
            'DAYS_REGISTRATION' : DAYS_REGISTRATION,
            'DAYS_ID_PUBLISH' : DAYS_ID_PUBLISH,
            'AMT_GOODS_PRICE' : AMT_GOODS_PRICE,
            'AMT_CREDIT' : AMT_CREDIT,
            'AMT_ANNUITY' : AMT_ANNUITY,
            'EXT_SOURCE_2' : EXT_SOURCE_2,
            'EXT_SOURCE_3' : EXT_SOURCE_3,
            'REGION_RATING_CLIENT_W_CITY' : REGION_RATING_CLIENT_W_CITY,
            'REG_CITY_NOT_LIVE_CITY' : REG_CITY_NOT_LIVE_CITY,
            'REG_CITY_NOT_WORK_CITY' : REG_CITY_NOT_WORK_CITY,
            'LIVE_CITY_NOT_WORK_CITY' :  LIVE_CITY_NOT_WORK_CITY,
           }
   
    # Creating DataFrame
    parameters = pd.DataFrame(data, index = [0])
    
    # Adding 
    parameters['CREDIT_INCOME_PERCENT'] = parameters['AMT_CREDIT'] / parameters['AMT_INCOME_TOTAL']
    
   # Returning DataFrame created
    return parameters

# df is the output of our function
df = user_input()

st.subheader('Your Data. Please check before validate!')
st.write(df)

########################################################################################################################

# Informations
if st.button('How does it work?'):
    st.write('This app is based on analysis of almost 300.000 loans past')
    st.write('To predict issue of your loan request, we use a machine learning algorithm')
    st.write('')
    st.write('We use all your input features and we add one important feature:')
    st.write('')
    st.write('- The amount of your credit / Your total income amount')
    st.write('')
    st.write('In other words, your input informations must respect some statistical values (according to our bank policy)')
    st.write('Just below we give you some statistics')
    st.write('When having completed your informations, you could either see result, or compare your informations with people in the same group as you (actually 12 groups existing)')

###########################################################################################################################
def stat_explanation():
    ''' This function returns explanations of descriptive statistics'''
    
    
    st.write('count : Number of persons in the group')
    st.write('mean : The mean for each feature')
    st.write('std : The stud for each feature (We can define the stud as the the mean of the deviations from the mean')
    st.write('Percentile (25%, 50%, 75%) : For exemple 25% : For each feature, 25% of people have a lower (or equal) value than the indicated value and 75% have an higher value') 
    st.write('min/max : The minimum/maximum value for each feature')



##############################################################################################################################
# Printing global statistics when clicking button
if st.button('Global statistics'):
    st.write(stat)
    stat_explanation()


##################################################################################################################
#Encoding AGE
def encode(x):
    '''This function encodes AGE'''
    
    
    if 20.0 <x<= 25.0:
        return 0
    elif 25.0 < x <= 30.0:
        return 1
    elif 30.0 < x <= 35.0:
        return 2
    elif 35.0 < x <= 40.0:
        return 3
    elif 40.0 < x <= 45.0:
        return 4
    elif 45.0 < x <= 50.0:
        return 5
    elif 50.0 < x <= 55.0:
        return 6
    elif 55.0 < x <= 60.0:
        return 7
    elif 60.0 < x <= 65.0:
        return 8
    else:
        return 9
    
    
# Apply    
df['AGE'] = df['AGE'].apply(encode)

###################################################################################################################


# According to our model we scale our data
# MinMaxScaler formula
# X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def scaling(df):

    

    df.loc[0, 'AMT_INCOME_TOTAL'] = (df.loc[0, 'AMT_INCOME_TOTAL'] - stat['AMT_INCOME_TOTAL']['min']) / (stat['AMT_INCOME_TOTAL']['max'] - stat['AMT_INCOME_TOTAL']['min']) 

    df.loc[0, 'DAYS_EMPLOYED'] = (df.loc[0, 'DAYS_EMPLOYED'] - stat['DAYS_EMPLOYED']['min']) / (stat['DAYS_EMPLOYED']['max'] - stat['DAYS_EMPLOYED']['min'])

    df.loc[0, 'DAYS_REGISTRATION'] = (df.loc[0, 'DAYS_REGISTRATION'] - stat['DAYS_REGISTRATION']['min']) / (stat['DAYS_REGISTRATION']['max'] - stat['DAYS_REGISTRATION']['min'])

    df.loc[0, 'DAYS_ID_PUBLISH'] = (df.loc[0, 'DAYS_ID_PUBLISH'] - stat['DAYS_ID_PUBLISH']['min']) / (stat['DAYS_ID_PUBLISH']['max'] - stat['DAYS_ID_PUBLISH']['min'])

    df.loc[0, 'AMT_GOODS_PRICE'] = (df.loc[0, 'AMT_GOODS_PRICE'] - stat['AMT_GOODS_PRICE']['min']) / (stat['AMT_GOODS_PRICE']['max'] - stat['AMT_GOODS_PRICE']['min'])

    df.loc[0, 'AMT_CREDIT'] = (df.loc[0, 'AMT_CREDIT'] - stat['AMT_CREDIT']['min']) / (stat['AMT_CREDIT']['max'] - stat['AMT_CREDIT']['min'] )

    df.loc[0, 'AMT_ANNUITY'] = (df.loc[0, 'AMT_ANNUITY'] - stat['AMT_ANNUITY']['min']) / (stat['AMT_ANNUITY']['max'] - stat['AMT_ANNUITY']['min'] )

    df.loc[0, 'EXT_SOURCE_2'] = (df.loc[0, 'EXT_SOURCE_2'] - stat['EXT_SOURCE_2']['min']) / (stat['EXT_SOURCE_2']['max'] - stat['EXT_SOURCE_2']['min'])

    df.loc[0, 'EXT_SOURCE_3'] = (df.loc[0, 'EXT_SOURCE_3'] - stat['EXT_SOURCE_3']['min']) / (stat['EXT_SOURCE_3']['max'] - stat['EXT_SOURCE_3']['min'])

    df.loc[0, 'CREDIT_INCOME_PERCENT'] = (df.loc[0, 'CREDIT_INCOME_PERCENT'] - stat['CREDIT_INCOME_PERCENT']['min']) / (stat['CREDIT_INCOME_PERCENT']['max'] - stat['CREDIT_INCOME_PERCENT']['min'] )

scaling(df)
###########################################################################################################################   
# OneHotEncoding df columns concerned
def onehot(df):
    
    if df.loc[0, 'REGION_RATING_CLIENT_W_CITY'] == 1:
    
        df['REGION_RATING_CLIENT_W_CITY_1'] = 1
        df['REGION_RATING_CLIENT_W_CITY_2'] = 0
        df['REGION_RATING_CLIENT_W_CITY_3'] = 0
                                    
        if df.loc[0, 'REG_CITY_NOT_LIVE_CITY'] == 0:
    
            df['REG_CITY_NOT_LIVE_CITY_0'] = 1
            df['REG_CITY_NOT_LIVE_CITY_1'] = 0
        else:
            df['REG_CITY_NOT_LIVE_CITY_0'] = 0
            df['REG_CITY_NOT_LIVE_CITY_1'] = 1
            

        if df.loc[0, 'REG_CITY_NOT_WORK_CITY'] == 0:
    
            df['REG_CITY_NOT_WORK_CITY_0'] = 1
            df['REG_CITY_NOT_WORK_CITY_1'] = 0
        else:
            df['REG_CITY_NOT_WORK_CITY_0'] = 0
            df['REG_CITY_NOT_WORK_CITY_1'] = 1
      
        if df.loc[0, 'LIVE_CITY_NOT_WORK_CITY'] == 0:
    
            df['LIVE_CITY_NOT_WORK_CITY_0'] = 1
            df['LIVE_CITY_NOT_WORK_CITY_1'] = 0
        else:
            df['LIVE_CITY_NOT_WORK_CITY_0'] = 0
            df['LIVE_CITY_NOT_WORK_CITY_1'] = 1
            
    elif  df.loc[0, 'REGION_RATING_CLIENT_W_CITY'] == 2:
        
        df['REGION_RATING_CLIENT_W_CITY_1'] = 0
        df['REGION_RATING_CLIENT_W_CITY_2'] = 1
        df['REGION_RATING_CLIENT_W_CITY_3'] = 0
        
        
        if df.loc[0, 'REG_CITY_NOT_LIVE_CITY'] == 0:
    
            df['REG_CITY_NOT_LIVE_CITY_0'] = 1
            df['REG_CITY_NOT_LIVE_CITY_1'] = 0
        else:
            df['REG_CITY_NOT_LIVE_CITY_0'] = 0
            df['REG_CITY_NOT_LIVE_CITY_1'] = 1


        if df.loc[0, 'REG_CITY_NOT_WORK_CITY'] == 0:
    
            df['REG_CITY_NOT_WORK_CITY_0'] = 1
            df['REG_CITY_NOT_WORK_CITY_1'] = 0
        else:
            df['REG_CITY_NOT_WORK_CITY_0'] = 0
            df['REG_CITY_NOT_WORK_CITY_1'] = 1
      
        if df.loc[0, 'LIVE_CITY_NOT_WORK_CITY'] == 0:
    
            df['LIVE_CITY_NOT_WORK_CITY_0'] = 1
            df['LIVE_CITY_NOT_WORK_CITY_1'] = 0
        else:
            df['LIVE_CITY_NOT_WORK_CITY_0'] = 0
            df['LIVE_CITY_NOT_WORK_CITY_1'] = 1
            
    elif df.loc[0, 'REGION_RATING_CLIENT_W_CITY'] == 3:
        
        df['REGION_RATING_CLIENT_W_CITY_1'] = 0
        df['REGION_RATING_CLIENT_W_CITY_2'] = 0
        df['REGION_RATING_CLIENT_W_CITY_3'] = 1
        
        if df.loc[0, 'REG_CITY_NOT_LIVE_CITY'] == 0:
    
            df['REG_CITY_NOT_LIVE_CITY_0'] = 1
            df['REG_CITY_NOT_LIVE_CITY_1'] = 0
        else:
            df['REG_CITY_NOT_LIVE_CITY_0'] = 0
            df['REG_CITY_NOT_LIVE_CITY_1'] = 1


        if df.loc[0, 'REG_CITY_NOT_WORK_CITY'] == 0:
    
            df['REG_CITY_NOT_WORK_CITY_0'] = 1
            df['REG_CITY_NOT_WORK_CITY_1'] = 0
        else:
            df['REG_CITY_NOT_WORK_CITY_0'] = 0
            df['REG_CITY_NOT_WORK_CITY_1'] = 1
      
        if df.loc[0, 'LIVE_CITY_NOT_WORK_CITY'] == 0:
    
            df['LIVE_CITY_NOT_WORK_CITY_0'] = 1
            df['LIVE_CITY_NOT_WORK_CITY_1'] = 0
        else:
            df['LIVE_CITY_NOT_WORK_CITY_0'] = 0
            df['LIVE_CITY_NOT_WORK_CITY_1'] = 1

    df.drop(['REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'], axis = 1, inplace = True)
#####################################################################################################################################
#Encoding AGE
def other_encode(x):
    '''This function encodes AGE'''
    
    
    if 20.0 <x<= 25.0:
        return 22.5
    elif 25.0 < x <= 30.0:
        return 27.5
    elif 30.0 < x <= 35.0:
        return 32.5
    elif 35.0 < x <= 40.0:
        return 37.5
    elif 40.0 < x <= 45.0:
        return 42.5
    elif 45.0 < x <= 50.0:
        return 47.5
    elif 50.0 < x <= 55.0:
        return 52.5
    elif 55.0 < x <= 60.0:
        return 57.5
    elif 60.0 < x <= 65.0:
        return 62.5
    else:
        return 67.5
    
#####################################################################################################################################


if st.sidebar.button('Validate/See results'):
    
    
    scaling(df)
    onehot(df)


    # Prediction
    prediction = XGB.predict(df)
    prob = XGB.predict_proba(df)
    
   
    
    if prediction == 0:
        st.write('ACCEPTED', prob)
        st.write("Under 0 you have the probability you won't be in defaut payment")
        st.write("Under 1 you have the probability you will be in defaut payment")
    else:
        st.write('REFUSED', prob)
        st.write("Under 0 you have the probability that you won't be in defaut payment")
        st.write("Under 1 you have the probability you will be in defaut payment")

    
        
if st.sidebar.button('Compare'):
     
    # Apply    
    df['AGE'] = df['AGE'].apply(other_encode)
    # Getting prediction
    pred = km.predict(df)
    
    if pred == 0:
        st.write('Satistics of the group you belong to:')
        st.write(stat0)
        stat_explanation()
        
    elif pred == 1:
        st.write('Satistics of the group you belong to:')
        st.write(stat1)
        stat_explanation()
        
    elif pred == 2:
        st.write('Satistics of the group you belong to:')
        st.write(stat2)
        stat_explanation()
        
    elif pred == 3:
        st.write('Satistics of the group you belong to:')
        st.write(stat3)
        stat_explanation()
        
    elif pred == 4:
        st.write('Satistics of the group you belong to:')
        st.write(stat4)
        stat_explanation()
        
    elif pred == 5:
        st.write('Satistics of the group you belong to:')
        st.write(stat5)
        stat_explanation()
        
    elif pred == 6:
        st.write('Satistics of the group you belong to:')
        st.write(stat6)
        stat_explanation()
        
    elif pred == 7:
        st.write('Satistics of the group you belong to:')
        st.write(stat7)
        stat_explanation()
        
   




