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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


st.set_option('deprecation.showPyplotGlobalUse', False)


# Files importations####################################################################################################

# 9 statistics table (9 clusters)

stat0 = pd.read_csv(os.path.join('.', 'stat0.csv'), index_col=0)
stat1 = pd.read_csv(os.path.join('.', 'stat1.csv'), index_col=0)
stat2 = pd.read_csv(os.path.join('.', 'stat2.csv'), index_col=0)
stat3 = pd.read_csv(os.path.join('.', 'stat3.csv'), index_col=0)
stat4 = pd.read_csv(os.path.join('.', 'stat4.csv'), index_col=0)
stat5 = pd.read_csv(os.path.join('.', 'stat5.csv'), index_col=0)
stat6 = pd.read_csv(os.path.join('.', 'stat6.csv'), index_col=0)
stat7 = pd.read_csv(os.path.join('.', 'stat7.csv'), index_col=0)
stat8 = pd.read_csv(os.path.join('.', 'stat8.csv'), index_col=0)

# 9 DataFrame for each cluster and the global datas


X = pd.read_csv(os.path.join('.', 'Xglobal.csv'), index_col=0)
X0 =  pd.read_csv(os.path.join('.', 'X0.csv'), index_col=0)
X1 =  pd.read_csv(os.path.join('.', 'X1.csv'), index_col=0)
X2 =  pd.read_csv(os.path.join('.', 'X2.csv'), index_col=0)
X3 =  pd.read_csv(os.path.join('.', 'X3.csv'), index_col=0)
X4 =  pd.read_csv(os.path.join('.', 'X4.csv'), index_col=0)
X5 =  pd.read_csv(os.path.join('.', 'X5.csv'), index_col=0)
X6 =  pd.read_csv(os.path.join('.', 'X6.csv'), index_col=0)
X7 =  pd.read_csv(os.path.join('.', 'X7.csv'), index_col=0)
X8 =  pd.read_csv(os.path.join('.', 'X8.csv'), index_col=0)





# load the model from disk
km = pickle.load (open ('clustering_9', 'rb'))
XGB = pickle.load (open ('XGBClassifier', 'rb'))
    

# Importing our global stats
stat = pd.read_csv(os.path.join('.', 'Xglobal_describe.csv'), index_col=0)

# Importing our X_train and y_train

y_train = pd.read_csv(os.path.join('.', 'y_train.csv'), index_col=0)
X_train = pd.read_csv(os.path.join('.', 'X_train.csv'), index_col=0)
############################################################################################################################

st.title('Prêt à dépenser BANK')

# title and sub-title
st.write('''
# Will your LOAN be approved?

''')
check1 = st.checkbox('How does it work?')
# Informations
if check1:
    st.write('This app is based on analysis of almost 300.000 loans past')
    st.write('To predict issue of your loan request, we use a machine learning algorithm')
    st.write('')
    st.write('We use all your input features and we add one important feature:')
    st.write('')
    st.write('- The amount of your credit / Your total income amount')
    st.write('')
    st.write('In other words, your input informations must respect some statistical values (according to our bank policy)')
    st.write('Just below we give you some statistics')
    st.write('When having completed your informations, you could either see result, or compare your informations with people in the same group as you (actually 9 groups existing)')


check2 = st.checkbox('Features importance')
# Explanation of features importance
if check2:
    st.write('Features are classified in order of importance for our model: ')
    st.write('\n')
    st.write('Extern source 3:  the higher it is, the better it is for the acceptance of credit')
    st.write('Extern source 2:  the higher it is, the better it is for the acceptance of credit')
    st.write('AGE:   the older you are, the better it is for the acceptance of credit')
    st.write('Last registration change:   the older it is, the less good it is for the acceptance of credit')
    st.write('Amount goods price:  the higher it is, the better it is for the acceptance of credit')
    st.write('If your contact adress is not the same as your working adress, it has a negative impact for the acceptance of credit ')
    st.write('Days employed:   the higher it is, the better it is for the acceptance of credit')
    st.write('Rating of your region:   3, 2, 1... The best for the acceptance of credit(in this order')
    st.write('If your permanent adress is the same as your contact adress, it has a negative impact for the accepatance of credit')
    st.write('Annuity amount:   the higher it is, the less good it is for the acceptance of credit')
    st.write('Last identity document change: the older it is, the less good it is for the acceptance of credit')
    st.write('If your permanent adress is the same as your working adress, it has a positive impact for the acceptance of credit')
    st.write('Annual income:   the higher it is, the better it is for the acceptance of credit')
    st.write('Credit amount:   The higher it is, the less good it is for the acceptance of your credit')
    st.write('Credit / Annual income:   The higher it is, the less good it is for the acceptance of your credit')
   

# Title on sidebar
st.sidebar.header('Your informations')
st.sidebar.write('Please respect the order!')



# Controling credit amount
def credit(col):
        if (stat['CREDIT_INCOME_PERCENT']['max'])*col > stat['AMT_CREDIT']['max']:
            value =  stat['AMT_CREDIT']['max']
        else :
            value = (stat['CREDIT_INCOME_PERCENT']['max'])*col
            return value



def user_input():
    '''Creating a function which generates a DataFrame, accordind to consumer's inputs '''
    
   
    
    # Controlling income
    AMT_INCOME_TOTAL = st.sidebar.number_input( 'Annual Income (Between {}$ and {}$)'.format(stat['AMT_INCOME_TOTAL']['min'],  round(stat['AMT_INCOME_TOTAL']['max'], 1)) , min_value = stat['AMT_INCOME_TOTAL']['min'] , max_value = stat['AMT_INCOME_TOTAL']['max'])
    
    # Credit controled by credit function
    AMT_CREDIT = st.sidebar.number_input('Credit Amount', min_value = stat['AMT_CREDIT']['min'], max_value = credit(AMT_INCOME_TOTAL))
    
    # Controling annuity
    AMT_ANNUITY = st.sidebar.number_input('Annuity Amount', min_value = AMT_INCOME_TOTAL*0.07, max_value = AMT_INCOME_TOTAL*1.32)
    
    # Controling good price
    AMT_GOODS_PRICE = st.sidebar.number_input('Good price Amount (Minimum {}$ and {}$ allowed)'.format(stat['AMT_GOODS_PRICE']['min'], stat['AMT_GOODS_PRICE']['max']), min_value = stat['AMT_GOODS_PRICE']['min'], max_value = stat['AMT_GOODS_PRICE']['max'])
    
    # Controling days employed
    DAYS_EMPLOYED = st.sidebar.number_input('Days employed (Between {} and {})'.format(int(stat['DAYS_EMPLOYED']['min']), int(stat['DAYS_EMPLOYED']['max'])),  min_value = int(stat['DAYS_EMPLOYED']['min']),  max_value = int(stat['DAYS_EMPLOYED']['max']))
    
    # Controlling days registration
    DAYS_REGISTRATION = st.sidebar.number_input('Last registration change (in days, between {} and {})'.format( int(stat['DAYS_REGISTRATION']['min']), int(stat['DAYS_REGISTRATION']['max'])), min_value = int(stat['DAYS_REGISTRATION']['min']), max_value = int(stat['DAYS_REGISTRATION']['max']))
    
    # Controling days ID publish
    DAYS_ID_PUBLISH = st.sidebar.number_input('Last Identity document change (in days, between {} and {})'.format(int(stat['DAYS_ID_PUBLISH']['min']), int(stat['DAYS_ID_PUBLISH']['max'])), min_value = int(stat['DAYS_ID_PUBLISH']['min']), max_value = int(stat['DAYS_ID_PUBLISH']['max']))
    
    # Controling both sources
    EXT_SOURCE_2 = st.sidebar.slider('Extern source 2', stat['EXT_SOURCE_2']['min'], stat['EXT_SOURCE_2']['max'], 0.53)
    EXT_SOURCE_3 = st.sidebar.slider('Extern source 3', stat['EXT_SOURCE_3']['min'], stat['EXT_SOURCE_3']['max'], 0.51)
    
    # Controling age
    AGE = st.sidebar.slider('Your age', 19, 69, 40)
    
    
    # Creating selectbox for rating region 
    values1 = ['<select>',1, 2, 3]
    default_ix_1 = values1.index(1)
    REGION_RATING_CLIENT_W_CITY = st.sidebar.selectbox('Rating of your region (taking city into account)', values1, index=default_ix_1)
    
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
#Encoding AGE
def encode(x):
    '''This function encodes AGE for model'''
    
    
    if 19<=x<= 25.0:
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

###########################################################################################################################

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

###########################################################################################################################
# OneHotEncoding df columns concerned
def onehot(df):
    
    
    
        
                                    
    if df.loc[0, 'REG_CITY_NOT_LIVE_CITY'] == 0:
        
    
            
        df['REG_CITY_NOT_LIVE_CITY_0'] = 1
        df['REG_CITY_NOT_LIVE_CITY_1'] = 0
        
            

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

    df.drop(['REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'], axis = 1, inplace = True)

###########################################################################################################################

def encode_stats(x):
    '''This function encodes AGE to plot stats'''
    
    
    if x == 0:
        return 22.5
    elif x == 1:
        return 27.5
    elif x == 2:
        return 32.5
    elif x == 3:
        return 37.5
    elif x == 4:
        return 42.5
    elif x == 5:
        return 47.5
    elif x == 6:
        return 52.5
    elif x == 7:
        return 57.5
    elif x == 8:
        return 62.5
    else:
        return 67.5
    
    

       
##########################################################################################################################
# Giving the same order as our model
def order(df):
    
     df = df[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED','DAYS_REGISTRATION',
              'DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT_W_CITY',  'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE', 'CREDIT_INCOME_PERCENT',
            'REG_CITY_NOT_LIVE_CITY_0', 'REG_CITY_NOT_LIVE_CITY_1',
            'REG_CITY_NOT_WORK_CITY_0', 'REG_CITY_NOT_WORK_CITY_1',
            'LIVE_CITY_NOT_WORK_CITY_0', 'LIVE_CITY_NOT_WORK_CITY_1']]



#################################################################################################################################    
# Printing global statistics when clicking button
checkbox1 = st.checkbox('Global Statistics:')


def print_stats(stat, X):
    

    selectbox1 = st.selectbox(label = 'Select a feature', options = df.columns)


    if selectbox1 == 'AMT_INCOME_TOTAL':    
    
        st.write('Minimum income: {}   /    Maximum income: {}   /      Your income: {}'.format(stat['AMT_INCOME_TOTAL']['min'], stat['AMT_INCOME_TOTAL']['max'], df.loc[0, 'AMT_INCOME_TOTAL']))
        st.write('Mean income:  {}'.format(round(stat['AMT_INCOME_TOTAL']['mean'], 2)))
        st.write('Median income (value which separate exactly in 2 the distribution):  {}'.format(round(stat['AMT_INCOME_TOTAL']['50%'], 2)))

    
    elif selectbox1 == 'AGE':
    
        # Apply    
        df['AGE'] = df['AGE'].apply(encode_stats)
        sns.histplot(data = X, x = 'AGE', kde = True)
        plt.axvline(x = df.loc[0, 'AGE'], ymin = -1, ymax= 2, c = 'r', linestyle = 'dashed')
   
        st.pyplot()
        st.write('AGE is divided in 10 categories:')
        st.write('\n')
        st.write('Between 19 and 24, your category is 22.5')
        st.write('Between 25 and 29, your category is 27.5')
        st.write('Between 30 and 34, your category is 32.5')
        st.write('Between 35 and 39, your category is 37.5')
        st.write('Between 40 and 44, your category is 42.5')
        st.write('Between 45 and 49, your category is 47.5')
        st.write('Between 50 and 54, your category is 52.5')
        st.write('Between 55 and 59, your category is 57.5')
        st.write('Between 60 and 64, your category is 62.5')
        st.write('Between 65 and 69, your category is 67.5')
        st.write('\n')
        st.write('The base of the Red Dashed line is your value')
        st.write('The number of persons having the same age is the high of the band')
    
    else: 
        sns.histplot(data = X, x = selectbox1, kde = True)
        plt.axvline(x= df.loc[0, selectbox1], ymin = -1, ymax= 2, c = 'r', linestyle = 'dashed')
    
        st.pyplot()
        st.write('The base of the Red Dashed line is your value')
        st.write('The number of persons having the same value is the high of the band')
if checkbox1:
      print_stats(stat, X)     
##############################################################################################################################



if st.sidebar.button('Validate/See results'):
    
    # Preprocessing
    scaling(df)
    onehot(df)
    
    # Replacing columns int the same order as our model
    order(df)
    
    # Prediction
    prediction = XGB.predict(df)
    prob = XGB.predict_proba(df)
    
   
    # Printing results
    if prediction == 0:
        st.write('ACCEPTED', prob)
        st.write("Under 0 you have the probability you won't be in defaut payment")
        st.write("Under 1 you have the probability you will be in defaut payment")
    else:
        st.write('REFUSED', prob)
        st.write("Under 0 you have the probability that you won't be in defaut payment")
        st.write("Under 1 you have the probability you will be in defaut payment")

    
        

checkbox2 = st.checkbox('Statistics of the group you belong to:')
    
    
if checkbox2:
    
   
    # Getting prediction
    pred = km.predict(df)
    
    if pred == 0:
            
        print_stats(stat0, X0)
        
    elif pred == 1:
            
        print_stats(stat1, X1)
            
    elif pred == 2:
            
        print_stats(stat2, X2)
        
    elif pred == 3:
            
        print_stats(stat3, X3)
            
        
    elif pred == 4:
            
        print_stats(stat4, X4)
            
        
    elif pred == 5:
            
        print_stats(stat5, X5)
        
    elif pred == 6:
            
        print_stats(stat6, X6)
        
    elif pred == 7:
            
        print_stats(stat7, X7)
            
    else:
            
        print_stats(stat8, X8)
            
        
   




