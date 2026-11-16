"""Pandas + scikit-learn version of the full analysis (no Spark needed). Produces outputs/results.json and charts.
Run from project root:  python src/run_pandas_analysis.py"""
import numpy as np, pandas as pd, json, os
os.makedirs('outputs',exist_ok=True)
month_names=['January','February','March','April','May','June','July','August','September','October','November','December']
df=pd.read_csv('data/hotel_bookings.csv')

raw=len(df)
# --- cleaning
df['country']=df['country'].fillna('Unknown')
df=df[~((df.adults+df.children)==0)]
df=df[df.adr>0]
df['total_nights']=df.stays_in_weekend_nights+df.stays_in_week_nights
df['revenue']=df.adr*df.total_nights
df['lead_bucket']=pd.cut(df.lead_time,[-1,7,30,90,180,1000],labels=['0-7','8-30','31-90','91-180','180+'])
clean=len(df)
R={'raw':raw,'clean':clean,'cols':df.shape[1]}
R['cancel_rate']=df.is_canceled.mean()*100
R['adr']=df[df.is_canceled==0].adr.mean()
R['avg_lead']=df.lead_time.mean(); R['avg_nights']=df.total_nights.mean()
R['rev']=df[df.is_canceled==0].revenue.sum()
g=df.groupby('hotel').agg(b=('is_canceled','size'),c=('is_canceled','mean'),adr=('adr','mean'),lead=('lead_time','mean'))
R['hotel']=g.to_dict('index')
months=month_names
mb=df.groupby(['arrival_date_month','hotel']).size().unstack().reindex(months)
madr=df[df.is_canceled==0].groupby(['arrival_date_month','hotel']).adr.mean().unstack().reindex(months)
R['peak_month']=mb.sum(axis=1).idxmax(); R['low_month']=mb.sum(axis=1).idxmin()
R['adr_peak']=madr.mean(axis=1).idxmax(); R['adr_peak_val']=madr.mean(axis=1).max(); R['adr_low_val']=madr.mean(axis=1).min(); R['adr_low']=madr.mean(axis=1).idxmin()
sg=df.groupby('market_segment').agg(b=('is_canceled','size'),c=('is_canceled','mean')).sort_values('b',ascending=False)
sg['share']=sg.b/sg.b.sum()*100; R['seg']=sg.to_dict('index')
lb=df.groupby('lead_bucket',observed=True).is_canceled.mean()*100; R['lead']=lb.to_dict()
dp=df.groupby('deposit_type').is_canceled.mean()*100; R['dep']=dp.to_dict()
ct=df.groupby('country').size().sort_values(ascending=False); R['countries']=(ct/ct.sum()*100).head(5).to_dict()
rp=df.groupby('is_repeated_guest').is_canceled.mean()*100; R['rep']=rp.to_dict()
# --- model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
feat=['hotel','lead_time','market_segment','deposit_type','is_repeated_guest','previous_cancellations','total_of_special_requests','required_car_parking_spaces','adr','total_nights','adults','children','arrival_date_month']
X=pd.get_dummies(df[feat],drop_first=False).astype(float); y=df.is_canceled
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
res={}
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(Xtr)
lr=LogisticRegression(max_iter=500).fit(sc.transform(Xtr),ytr); pl=lr.predict(sc.transform(Xte)); prl=lr.predict_proba(sc.transform(Xte))[:,1]
rf=RandomForestClassifier(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42).fit(Xtr,ytr); pr=rf.predict(Xte); prr=rf.predict_proba(Xte)[:,1]
for n,pp,pb in [('Logistic Regression',pl,prl),('Random Forest',pr,prr)]:
    res[n]=dict(acc=accuracy_score(yte,pp)*100,prec=precision_score(yte,pp)*100,rec=recall_score(yte,pp)*100,f1=f1_score(yte,pp)*100,auc=roc_auc_score(yte,pb))
R['models']=res; R['train']=len(Xtr); R['test']=len(Xte)
cm=confusion_matrix(yte,pr); R['cm']=cm.tolist()
imp=pd.Series(rf.feature_importances_,index=X.columns)
# aggregate dummies
agg={}
for c in X.columns:
    k=next(f for f in feat if c==f or c.startswith(f+'_')); agg[k]=agg.get(k,0)+imp[c]
agg=pd.Series(agg).sort_values(ascending=False); R['imp']=agg.head(6).to_dict()
json.dump(R,open('outputs/results.json','w'),indent=1,default=float)
# --- charts
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Liberation Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
C1,C2,C3='#1f4e79','#e07b39','#5b9aa0'
def save(f,name): f.tight_layout(); f.savefig(name,dpi=170); plt.close(f)
# fig5.1
f,ax=plt.subplots(1,2,figsize=(9,4))
ax[0].bar(g.index,g.b,color=[C1,C2]); ax[0].set_title('Bookings by hotel type'); ax[0].set_ylabel('Number of bookings')
for i,v in enumerate(g.b): ax[0].text(i,v,f'{v:,}',ha='center',va='bottom')
ax[1].bar(g.index,g.c*100,color=[C1,C2]); ax[1].set_title('Cancellation rate by hotel type'); ax[1].set_ylabel('Cancellation rate (%)')
for i,v in enumerate(g.c*100): ax[1].text(i,v,f'{v:.1f}%',ha='center',va='bottom')
save(f,'outputs/fig5_1.png')
# fig5.2
f,ax=plt.subplots(1,2,figsize=(11,4)); x=np.arange(12); ab=[m[:3] for m in months]
w=.4
ax[0].bar(x-w/2,mb['City Hotel'],w,label='City Hotel',color=C1); ax[0].bar(x+w/2,mb['Resort Hotel'],w,label='Resort Hotel',color=C2)
ax[0].set_xticks(x); ax[0].set_xticklabels(ab); ax[0].set_title('Monthly bookings'); ax[0].legend(frameon=False)
ax[1].plot(ab,madr['City Hotel'],marker='o',color=C1,label='City Hotel'); ax[1].plot(ab,madr['Resort Hotel'],marker='o',color=C2,label='Resort Hotel')
ax[1].set_title('Average daily rate (ADR) by month'); ax[1].set_ylabel('ADR (EUR)'); ax[1].legend(frameon=False)
save(f,'outputs/fig5_2.png')
# fig5.3
f,ax=plt.subplots(1,2,figsize=(11,4.2))
s2=sg.sort_values('b'); ax[0].barh(s2.index,s2.share,color=C3); ax[0].set_title('Market segment share of bookings'); ax[0].set_xlabel('Share (%)')
ax[1].plot(list(lb.index),lb.values,marker='o',color=C2,lw=2); ax[1].set_title('Cancellation rate by lead time'); ax[1].set_xlabel('Lead time (days)'); ax[1].set_ylabel('Cancellation rate (%)')
save(f,'outputs/fig5_3.png')
# fig5.4
f,ax=plt.subplots(1,2,figsize=(10.5,4.2))
a2=agg.head(8)[::-1]; ax[0].barh(a2.index,a2.values,color=C1); ax[0].set_title('Random Forest feature importance')
im=ax[1].imshow(cm,cmap='Blues'); ax[1].set_xticks([0,1]); ax[1].set_yticks([0,1]); ax[1].set_xticklabels(['Not canceled','Canceled']); ax[1].set_yticklabels(['Not canceled','Canceled'])
ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Actual'); ax[1].set_title('Confusion matrix (test set)')
for i in range(2):
    for j in range(2): ax[1].text(j,i,f'{cm[i][j]:,}',ha='center',va='center',color='white' if cm[i][j]>cm.max()/2 else 'black')
ax[1].spines[:].set_visible(False)
save(f,'outputs/fig5_4.png')
# fig5.5 dashboard
f=plt.figure(figsize=(12,6.4)); gs=f.add_gridspec(2,3,hspace=.5,wspace=.35)
ax=f.add_subplot(gs[0,0]); ax.bar(g.index,g.b,color=[C1,C2]); ax.set_title('Bookings by hotel')
ax=f.add_subplot(gs[0,1]); ax.plot(ab,mb.sum(axis=1),marker='o',color=C1); ax.set_title('Bookings per month'); ax.tick_params(axis='x',labelsize=8)
ax=f.add_subplot(gs[0,2]); ax.plot(ab,madr.mean(axis=1),marker='o',color=C2); ax.set_title('Average ADR per month (EUR)'); ax.tick_params(axis='x',labelsize=8)
ax=f.add_subplot(gs[1,0]); ax.barh(s2.index,s2.share,color=C3); ax.set_title('Segment share (%)'); ax.tick_params(axis='y',labelsize=8)
ax=f.add_subplot(gs[1,1]); ax.bar(dp.index,dp.values,color=[C1,C2,C3]); ax.set_title('Cancellation % by deposit type'); ax.tick_params(axis='x',labelsize=8)
ax=f.add_subplot(gs[1,2]); n=list(res); ax.bar(n,[res[k]['acc'] for k in n],color=[C3,C1]); ax.set_ylim(0,100); ax.set_title('Model accuracy (%)'); ax.tick_params(axis='x',labelsize=8)
for i,k in enumerate(n): ax.text(i,res[k]['acc'],f"{res[k]['acc']:.1f}",ha='center',va='bottom')
save(f,'outputs/fig5_5.png')
print(json.dumps(R,indent=1,default=float))
