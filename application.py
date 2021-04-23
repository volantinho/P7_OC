# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:58:41 2021

@author: VOLANTE
"""

import pandas as pd
import streamlit as st
import numpy as np

# Importing our best model for prediction
from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(max_depth = 8, max_features = 'sqrt',
                           min_samples_leaf = 50, min_samples_split = 1000,
                           random_state = 10, subsample = 0.8)

# Importing our data_stats
stat =  pd.read_csv('C:/Users/VOLANTE/anaconda3/envs/OC/P7/X_stats.csv', index_col = 0)

# Importing our X_train and y_train
y_train = pd.read_csv('C:/Users/VOLANTE/anaconda3/envs/OC/P7/y_train.csv', index_col = 0)
X_train = pd.read_csv('C:/Users/VOLANTE/anaconda3/envs/OC/P7/X_train.csv', index_col =0)

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
    CNT_CHILDREN = st.sidebar.slider('Number of children', 0, 19, 1) 
    AMT_INCOME_TOTAL = st.sidebar.number_input( 'Annual Income (Between {}$ and {}$)'.format(stat['AMT_INCOME_TOTAL']['min'],  stat['AMT_INCOME_TOTAL']['max']) , min_value = stat['AMT_INCOME_TOTAL']['min'] , max_value = stat['AMT_INCOME_TOTAL']['max'])
    DAYS_EMPLOYED = st.sidebar.number_input('Days employed (Between {} and {}'.format(int(stat['DAYS_EMPLOYED']['min']), int(stat['DAYS_EMPLOYED']['max'])),  min_value = int(stat['DAYS_EMPLOYED']['min']),  max_value = int(stat['DAYS_EMPLOYED']['max']))
    DAYS_REGISTRATION = st.sidebar.number_input('Last registration change (in days, between {} and {})'.format( int(stat['DAYS_REGISTRATION']['min']), int(stat['DAYS_REGISTRATION']['max'])), min_value = int(stat['DAYS_REGISTRATION']['min']), max_value = int(stat['DAYS_REGISTRATION']['max']))
    DAYS_ID_PUBLISH = st.sidebar.number_input('Last Identity document change (in days, between {} and {})'.format(int(stat['DAYS_ID_PUBLISH']['min']), int(stat['DAYS_ID_PUBLISH']['max'])), min_value = int(stat['DAYS_ID_PUBLISH']['min']), max_value = int(stat['DAYS_ID_PUBLISH']['max']))
    AMT_GOODS_PRICE = st.sidebar.number_input('Good price Amount (Between {}$ and {}$ allowed)'.format(stat['AMT_GOODS_PRICE']['min'], stat['AMT_GOODS_PRICE']['max']), min_value = stat['AMT_GOODS_PRICE']['min'], max_value = stat['AMT_GOODS_PRICE']['max'])
    AMT_CREDIT = st.sidebar.number_input('Credit Amount', min_value = stat['AMT_CREDIT']['min'], max_value = (stat['CREDIT_GOOD_PERCENT']['max']) * AMT_GOODS_PRICE)
    AMT_ANNUITY = st.sidebar.number_input('Annuity Amount', min_value = (stat['CREDIT_ANNUITY_PERCENT']['min'])*AMT_CREDIT, max_value = (stat['ANNUITY_INCOME_PERCENT']['max'])*AMT_INCOME_TOTAL)
    EXT_SOURCE_2 = st.sidebar.slider('Extern source 2', stat['EXT_SOURCE_2']['min'], stat['EXT_SOURCE_2']['max'], 0.53)
    EXT_SOURCE_3 = st.sidebar.slider('Extern source 3', stat['EXT_SOURCE_3']['min'], stat['EXT_SOURCE_3']['max'], 0.51)
    
    
    
    # Creating a dictionary
    data = {'AGE' : AGE,
            'CNT_CHILDREN' : CNT_CHILDREN,
            'AMT_INCOME_TOTAL' : AMT_INCOME_TOTAL,
            'DAYS_EMPLOYED' : DAYS_EMPLOYED,
            'DAYS_REGISTRATION' : DAYS_REGISTRATION,
            'DAYS_ID_PUBLISH' : DAYS_ID_PUBLISH,
            'AMT_GOODS_PRICE' : AMT_GOODS_PRICE,
            'AMT_CREDIT' : AMT_CREDIT,
            'AMT_ANNUITY' : AMT_ANNUITY,
            'EXT_SOURCE_2' : EXT_SOURCE_2,
            'EXT_SOURCE_3' : EXT_SOURCE_3,
            
           }
   
    # Creating DataFrame
    parameters = pd.DataFrame(data, index = [0])
    
    # Returning DataFrame created
    return parameters

# df is the output of our function
df = user_input()

st.subheader('Your Data. Please check before validate!')
st.write(df)

########################################################################################################################

# Informations
if st.button('How does it work?'):
    st.write('This app is based on analysis of more than 200.000 loans past')
    st.write('To predict issue of your loan request, we use a machine learning algorithm')
    st.write('')
    st.write('We use all your input features and we add 4 more important features:')
    st.write('')
    st.write('- The amount of your credit / Your total income amount')
    st.write('- The amount of your annuity / Your total income amount')
    st.write('- The amount of your annuity / Your credit amount')
    st.write('- The amount of your credit / Your good price amount')
    st.write('')
    st.write('In other words, your input informations must respect some statistical values (according to our bank policy)')
    st.write('Just below we give you some stats')

##############################################################################################################################
# Printing global statistics when clickin button
if st.button('Global statistics'):
    
        
    st.write('Age :    minimum: {}    maximum: {}    median: {}'.format(21, 69, 40))
    st.write('Number of children :    minimum: {}    maximum: {}    median: {}'.format(stat['CNT_CHILDREN']['min'], stat['CNT_CHILDREN']['max'], stat['CNT_CHILDREN']['50%']))
    st.write('Annual income :    minimum: {}    maximum: {}    median: {}'.format(stat['AMT_INCOME_TOTAL']['min'], stat['AMT_INCOME_TOTAL']['max'], stat['AMT_INCOME_TOTAL']['50%']))
    st.write('Days employed :    minimum: {}    maximum: {}    median: {}'.format(stat['DAYS_EMPLOYED']['min'], stat['DAYS_EMPLOYED']['max'], stat['DAYS_EMPLOYED']['50%']))
    st.write('Last registration change :    minimum: {}    maximum: {}    median: {}'.format(stat['DAYS_REGISTRATION']['min'], stat['DAYS_REGISTRATION']['max'], stat['DAYS_REGISTRATION']['50%']))
    st.write('Last Identity document change :    minimum: {}    maximum: {}    median: {}'.format(round(stat['DAYS_ID_PUBLISH']['min'], 2), round(stat['DAYS_ID_PUBLISH']['max'], 2), round(stat['DAYS_ID_PUBLISH']['50%'], 2)))
    st.write('Good price Amount :    minimum: {}    maximum: {}    median: {}'.format(round(stat['AMT_GOODS_PRICE']['min'], 2), round(stat['AMT_GOODS_PRICE']['max'], 2), round(stat['AMT_GOODS_PRICE']['50%'], 2)))
    st.write('Credit Amount :    minimum: {}    maximum: {}    median: {}'.format(round(stat['AMT_CREDIT']['min'], 2), round(stat['AMT_CREDIT']['max'], 2), round(stat['AMT_CREDIT']['50%'], 2)))
    st.write('Annuity Amount:    minimum: {}    maximum: {}    median: {}'.format(stat['AMT_ANNUITY']['min'], stat['AMT_ANNUITY']['max'], stat['AMT_ANNUITY']['50%']))
    st.write('Extern source 2 :    minimum: {}    maximum: {}    median: {}'.format(round(stat['EXT_SOURCE_2']['min'], 2), round(stat['EXT_SOURCE_2']['max'], 2), round(stat['EXT_SOURCE_2']['50%'], 2)))
    st.write('Extern source 3 :    minimum: {}    maximum: {}    median: {}'.format(round(stat['EXT_SOURCE_3']['min'], 2), round(stat['EXT_SOURCE_3']['max'], 2), round(stat['EXT_SOURCE_3']['50%'], 2)))
    st.write('Amount Credit / Annual income :    minimum: {}    maximum: {}    median: {}'.format(round(stat['CREDIT_INCOME_PERCENT']['min'], 2), round(stat['CREDIT_INCOME_PERCENT']['max'], 2), round(stat['CREDIT_INCOME_PERCENT']['50%'], 2)))
    st.write('Annuity amount / Annual amount :    minimum: {}    maximum: {}    median: {}'.format(round(stat['ANNUITY_INCOME_PERCENT']['min'], 2), round(stat['ANNUITY_INCOME_PERCENT']['max'], 2), round(stat['ANNUITY_INCOME_PERCENT']['50%'], 2)))
    st.write('Annuity amount / Credit amount :    minimum: {}    maximum: {}    median: {}'.format(round(stat['CREDIT_ANNUITY_PERCENT']['min'], 2), round(stat['CREDIT_ANNUITY_PERCENT']['max'], 2), round(stat['CREDIT_ANNUITY_PERCENT']['50%'], 2)))
    st.write('Amount credit / Good price amount :    minimum: {}    maximum: {}    median: {}'.format(round(stat['CREDIT_GOOD_PERCENT']['min'], 2), round(stat['CREDIT_GOOD_PERCENT']['max'], 2), round(stat['CREDIT_GOOD_PERCENT']['50%'], 2)))
    
        
        
##################################################################################################################
def adding_columns(df):

    # Creating our 4 other features for model
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_ANNUITY_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CREDIT_GOOD_PERCENT'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']  


##################################################################################################################
#Encoding AGE
def encode(x):
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
    
    
    
df['AGE'] = df['AGE'].apply(encode)

###################################################################################################################



# MinMaxScaler formula
# X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def scaling(df):

    df.loc[0, 'CNT_CHILDREN'] = (df.loc[0, 'CNT_CHILDREN'] - stat['CNT_CHILDREN']['min']) / (stat['CNT_CHILDREN']['max'] - stat['CNT_CHILDREN']['min'] )

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

    df.loc[0, 'ANNUITY_INCOME_PERCENT'] = (df.loc[0, 'ANNUITY_INCOME_PERCENT'] - stat['ANNUITY_INCOME_PERCENT']['min']) / (stat['ANNUITY_INCOME_PERCENT']['max'] - stat['ANNUITY_INCOME_PERCENT']['min'])

    df.loc[0, 'CREDIT_ANNUITY_PERCENT'] = (df.loc[0, 'CREDIT_ANNUITY_PERCENT'] - stat['CREDIT_ANNUITY_PERCENT']['min']) / (stat['CREDIT_ANNUITY_PERCENT']['max'] - stat['CREDIT_ANNUITY_PERCENT']['min'])

    df.loc[0, 'CREDIT_GOOD_PERCENT'] = (df.loc[0, 'CREDIT_GOOD_PERCENT'] - stat['CREDIT_GOOD_PERCENT']['min']) / (stat['CREDIT_GOOD_PERCENT']['max'] - stat['CREDIT_GOOD_PERCENT']['min'] )


####################################################################################################################


def preprocessing (X_train):
    # Preprocessing
    from sklearn.preprocessing import MinMaxScaler

    # Giving 0 for min and 1 for max.. for the rest the proportion
    scaler = MinMaxScaler()

    #Scaling X_train excepted  'age'
    for col in X_train.drop('AGE', axis = 1).columns :
        X_train.loc[:, col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    
    # Encoding 'AGE'
    from sklearn.preprocessing import LabelEncoder
    
    # Creating OneHotEncoder
    encoder = LabelEncoder()

    # Encoding 'AGE' for X_train
    X_train['AGE'] = encoder.fit_transform(X_train['AGE'].values.reshape(-1, 1))



####################################################################################################################


if st.sidebar.button('Validate/See results'):
    
    adding_columns(df)
    scaling(df)
    preprocessing(X_train)
    
    # Converting in arrays numpy
    df = np.array(df)
    X_train =  np.array(X_train)
    y_train =  np.array(y_train)

    # Fitting model
    GB.fit(X_train, y_train)

    
    #import pdb;pdb.set_trace()
    # Prediction
    prediction = GB.predict(df)
    prob = GB.predict_proba(df)
    
   
    
    if prediction == 0:
        st.write('ACCEPTED', prob)
        st.write("Under 0 you have the probability you won't be in defaut payment")
        st.write("Under 1 you have the probability you will be in defaut payment")
    else:
        st.write('REFUSED', prob)
        st.write("Under 0 you have the probability that you won't be in defaut payment")
        st.write("Under 1 you have the probability you will be in defaut payment")

 
        




