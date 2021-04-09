# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:58:41 2021

@author: VOLANTE
"""

import pandas as pd
import streamlit as st



st.title('Prêt à dépenser BANK')

# title and sub-title
st.write('''
# Will your LOAN be approved?

''')

# Title on sidebar
st.sidebar.header('Your informations')



def user_input():
    '''Creating a function which generates a DataFrame, accordind to consumer's inputs '''
    
    AGE = st.sidebar.slider('Your age', 21, 69, 40)
    CNT_CHILDREN = st.sidebar.slider('Number of children', 0, 19, 1)
    AMT_INCOME_TOTAL = st.sidebar.number_input( 'Annual Income (allowed between 26550$ and 9M$)' , min_value = 26550 , max_value = 9000000)
    DAYS_EMPLOYED = st.sidebar.slider('Days employed', 0, 17912, 2532)
    DAYS_REGISTRATION = st.sidebar.number_input('Last registration change (in days)', min_value = 0, max_value = 22701)
    DAYS_ID_PUBLISH = st.sidebar.slider('Last Identity document change (in days)', 0, 7197, 2884)
    AMT_GOODS_PRICE = st.sidebar.number_input('Good price Amount (between 40500$ and 4.05M$)', min_value = 40500, max_value = 4050000)
    AMT_CREDIT = st.sidebar.number_input('Credit Amount(allowed between 45000$ and 4.05M$)', min_value = 45000, max_value = 4050000)
    AMT_ANNUITY = st.sidebar.number_input('Annuity Amount (between 1980$ and 258026$)', 1980, 258026, 27986)
    EXT_SOURCE_2 = st.sidebar.slider('Extern source 2', 0.0, 1.0, 0.53)
    EXT_SOURCE_3 = st.sidebar.slider('Extern source 3', 0.0, 1.0, 0.51)
    
    
    
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

if st.checkbox('See results'):
     #Printing inputs results
     st.subheader('Your Data')
     st.write(df)

##################################################################################################################
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



####################################################################################################################
# Importing our final dataframe used for ML

X_train = pd.read_csv('C:/Users/VOLANTE/Desktop/OPEN PYTHON/projet 7/X_train_credit_final.csv', index_col = 0)

y_train = pd.read_csv('C:/Users/VOLANTE/Desktop/OPEN PYTHON/projet 7/y_train_credit_final.csv', index_col = 0)



# Importing our best model
from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(max_depth = 8, max_features = 'sqrt',
                           min_samples_leaf = 50, min_samples_split = 1000,
                           random_state = 10, subsample = 0.8)


####################################################################################################################


des = X_train.describe()



# Fitting model
GB.fit(X_train, y_train)

print(df)
# Prediction
prediction = GB.predict(df)

# Printing result for consumer
if prediction == 0:
    st.write('ACCEPTED')
else :
    st.write('REFUSED')
    













