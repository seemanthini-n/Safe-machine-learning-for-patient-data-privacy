# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:11:01 2019

@author: mansi
"""

import random
import pyffx

df_2 = pandas.read_csv('C:\\Users\\mansi\\OneDrive\\Documents\\ADM\\enc_data.csv' )

#reading all columns into data frame
col= df_2.iloc[:,10]
col1 = df_2.iloc[:,1]
col2 = df_2.iloc[:,2]
col3 = df_2.iloc[:,3]
col4 = df_2.iloc[:,4]
col5 = df_2.iloc[:,5]
col6 = df_2.iloc[:,6]
col7 = df_2.iloc[:,7]
col8 = df_2.iloc[:,8]
col9 = df_2.iloc[:,9]


e = pyffx.Integer(b'secret-key', length=5)

#using key encryption(format preserving)
encrypted_number_list1 = [e.encrypt(x) for x in col1]
encrypted_number_list2 = [e.encrypt(x) for x in col2]
encrypted_number_list3 = [e.encrypt(x) for x in col3]
encrypted_number_list4 = [e.encrypt(x) for x in col4]
encrypted_number_list5 = [e.encrypt(x) for x in col5]
encrypted_number_list6 = [e.encrypt(x) for x in col6]
encrypted_number_list7 = [e.encrypt(x) for x in col7]
encrypted_number_list8 = [e.encrypt(x) for x in col8]
encrypted_number_list9 = [e.encrypt(x) for x in col9]




from phe import paillier

public_key,private_key= paillier.generate_paillier_keypair(n_length=58)

#using public key homomorphic encryption of each column
df1 = [public_key.raw_encrypt(x) for x in encrypted_number_list1]
df2 = [public_key.raw_encrypt(x) for x in encrypted_number_list2]
df3 = [public_key.raw_encrypt(x) for x in encrypted_number_list3]
df4 = [public_key.raw_encrypt(x) for x in encrypted_number_list4]
df5 = [public_key.raw_encrypt(x) for x in encrypted_number_list5]
df6 = [public_key.raw_encrypt(x) for x in encrypted_number_list6]
df7 = [public_key.raw_encrypt(x) for x in encrypted_number_list7]
df8 = [public_key.raw_encrypt(x) for x in encrypted_number_list8]
df9 = [public_key.raw_encrypt(x) for x in encrypted_number_list9]

print(len(df1))
import numpy as np
import pandas as pd

result=df1+df2

#renaming of columns and taking list indo data frame
r1 = pd.DataFrame(np.array(df1).reshape(699), columns = list("a"))

r2 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("b"))
r3 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("c"))
r4 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("d"))
r5 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("e"))
r6 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("f"))
r7 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("g"))
r8 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("h"))
r9 = pd.DataFrame(np.array(df1).reshape(699,1), columns = list("i"))

#combining all data frames into one for writing into file
result= pd.concat([r1, r2,r3,r4,r5,r6,r7,r8,r9,col], axis=1)

result.to_csv('C:\\Users\\mansi\\OneDrive\\Documents\\ADM\\encrypted_data.csv', index=True)
