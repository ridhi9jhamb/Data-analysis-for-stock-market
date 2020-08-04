import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
AXJO.set_index(pd.to_datetime(AXJO['Date']),inplace=True,drop=True)
crude.set_index(pd.to_datetime(crude['Date']),inplace=True,drop=True)
euro.set_index(pd.to_datetime(euro['Date']),inplace=True,drop=True)
FTSE_L.set_index(pd.to_datetime(FTSE_L['Date']),inplace=True,drop=True)
HSI.set_index(pd.to_datetime(HSI['Date']),inplace=True,drop=True)
niftyenergy.set_index(pd.to_datetime(niftyenergy['Date']),inplace=True,drop=True)
niftyinfra.set_index(pd.to_datetime(niftyinfra['Date']),inplace=True,drop=True)
niftymetal.set_index(pd.to_datetime(niftymetal['Date']),inplace=True,drop=True)
niftypharma.set_index(pd.to_datetime(niftypharma['Date']),inplace=True,drop=True)
sandp.set_index(pd.to_datetime(sandp['Date']),inplace=True,drop=True)
silver.set_index(pd.to_datetime(silver['Date']),inplace=True,drop=True)
usd.set_index(pd.to_datetime(usd['Date']),inplace=True,drop=True)
nsebank.set_index(pd.to_datetime(nsebank['Date']),inplace=True,drop=True)
FTSE_L.drop_duplicates(inplace=True)


market=pd.DataFrame({'AXJO':AXJO['Open'],'HSI':HSI['Open'],'FTSE':FTSE_L['Open'],'S&P500':sandp['Open'],'USD':usd['Open'],'Crude':crude['Price'],'EURO':euro['Open'],'Silver':silver['Open'],'NiftyMetal':niftymetal['Open'],'NiftyInfra':niftyinfra['Open'],'NiftyEnergy':niftyenergy['Open'],'NiftyPharma':niftypharma['Open'],'NSEBANK':nsebank['Close']})
AXJO=pd.read_csv('AXJO_AUD.csv')
crude=pd.read_csv('CRUDE.csv')
euro=pd.read_csv('EURO.csv')
FTSE_L=pd.read_csv('FTSE_LONDON.csv')
HSI=pd.read_csv('HSI.csv')
niftyenergy=pd.read_csv('NIFTYENERGY.csv')
niftyinfra=pd.read_csv('NIFTYINFRA.csv')
niftymetal=pd.read_csv('NIFTYMETAL.csv')
niftypharma=pd.read_csv('NIFTYPHARMA.csv')
nsebank=pd.read_csv('NSEBANK.csv')
sandp=pd.read_csv('S&P 500_USD.csv')
silver=pd.read_csv('Silver.csv')
usd=pd.read_csv('USD.csv')

AXJO=pd.read_csv('AXJO_AUD.csv')
crude=pd.read_csv('CRUDE.csv')
euro=pd.read_csv('EURO.csv')
FTSE_L=pd.read_csv('FTSE_LONDON.csv')
HSI=pd.read_csv('HSI.csv')
niftyenergy=pd.read_csv('NIFTYENERGY.csv')
niftyinfra=pd.read_csv('NIFTYINFRA.csv')
niftymetal=pd.read_csv('NIFTYMETAL.csv')
niftypharma=pd.read_csv('NIFTYPHARMA.csv')
nsebank=pd.read_csv('NSEBANK.csv')
sandp=pd.read_csv('S&P 500_USD.csv')
silver=pd.read_csv('Silver.csv')
usd=pd.read_csv('USD.csv')

type(mark['Crude'][0])
lst=[]
#int(mark['Crude'][0].replace(',', ''))
for i in range(85):
    lst.append(int(mark['Crude'][i].replace(',','')))

mark.corr()
mark.corr()['NSEBANK'].sort_values(ascending=False)
from sklearn.preprocessing import StandardScaler
x = mark[['AXJO','HSI','FTSE','S&P500','USD','EURO','Silver','NiftyMetal','NiftyInfra','NiftyEnergy','NiftyPharma']].values
# Separating out the target
y = mark['NSEBANK'].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=11)
principalComponents = pca.fit_transform(x)
pca.explained_variance_ratio_*100
mark['short_avg'] = mark['NSEBANK'].rolling(window=5,min_periods=1).mean()
mark['long_avg'] = mark['NSEBANK'].rolling(window=30,min_periods=1).mean()
l=[]
for i in range(len(mark["NSEBANK"])):
    if mark['short_avg'][i]>mark['long_avg'][i]:
        l.append(1)
    else:
        l.append(0)
mark['label']=l
mark.iloc[70:85]
lf=LogisticRegression()
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
p=mark.iloc[0:73]
X=np.stack([p['NiftyInfra'], p['NiftyEnergy'],p['S&P500'], p['NiftyMetal'], p['FTSE'], p['AXJO'], p['HSI']], axis=1)


#x_train=mark.loc[0:73]
#df=mark[mark['label']==1]
#df1=mark[mark['label']==0]
y=mark.iloc[0:73]["label"]
lf.fit(X,y)
forest_classifier.fit(X,y)
p1=mark.iloc[74:85]
x_test=np.stack([p1['NiftyInfra'], p1['NiftyEnergy'],p1['S&P500'], p1['NiftyMetal'], p1['FTSE'], p1['AXJO'], p1['HSI']], axis=1)
y_test=mark.iloc[74:85]["label"]
lf.predict(x_test)
forest_classifier.predict(x_test)
from sklearn.metrics import accuracy_score
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)
accuracy(lf,X,y)
accuracy(forest_classifier,X,y)
lf.score(x_test,y_test)
forest_classifier.score(x_test,y_test)
