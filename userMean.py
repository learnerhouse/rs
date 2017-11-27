import pandas as pd
import numpy as np
# from numpy import linalg as la
#
# A = np.mat([[1, 2, 3], [4, 5, 6]])
# U,sigma,VT=la.svd(A)
# print (U)
test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')
rate_rank = train.groupby('uid').mean().loc[:,['score']].iloc[:,-1]
rate_rank=pd.DataFrame(np.int32((rate_rank*2).values),index=rate_rank.index,columns=['group'])
rate_rank_des = rate_rank.reset_index()

res = pd.merge(train,rate_rank_des,how='left',on='uid').groupby(['iid','group']).mean().reset_index().loc[:,['iid','group','score']]
test_plus = pd.merge(test,rate_rank_des,how='left',on='uid')
result6 = pd.merge(test_plus,res,how='left',on=['iid','group']).fillna(3.0)

