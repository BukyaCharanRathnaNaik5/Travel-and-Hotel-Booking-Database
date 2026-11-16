"""Generates a synthetic hotel booking dataset (same schema as the public Hotel Booking Demand dataset).
Run from the project root:  python src/generate_data.py
Skip this if you use the real Kaggle CSV: just put it at data/hotel_bookings.csv."""
import numpy as np, pandas as pd, json
rng=np.random.default_rng(42)
N=120000
hotel=rng.choice(['City Hotel','Resort Hotel'],N,p=[0.66,0.34])
month_names=['January','February','March','April','May','June','July','August','September','October','November','December']
city_w=np.array([5,6,8,9,10,10,10,11,9,8,5,5],float); res_w=np.array([6,7,7,8,8,8,11,12,8,7,5,6],float)
month=np.array([rng.choice(12,p=(city_w if h=='City Hotel' else res_w)/(city_w if h=='City Hotel' else res_w).sum()) for h in hotel])
seg=rng.choice(['Online TA','Offline TA/TO','Direct','Groups','Corporate','Complementary','Aviation'],N,p=[.47,.20,.105,.17,.045,.006,.004])
lead=np.clip(rng.gamma(1.3,70,N)+np.where(seg=='Groups',60,0)-np.where(seg=='Direct',25,0),0,600).astype(int)
deposit=np.where(seg=='Groups',rng.choice(['No Deposit','Non Refund','Refundable'],N,p=[.62,.37,.01]),rng.choice(['No Deposit','Non Refund','Refundable'],N,p=[.9,.095,.005]))
repeat=(rng.random(N)<np.where(seg=='Direct',.12,.02)).astype(int)
prev_c=np.where(rng.random(N)<.05,rng.integers(1,4,N),0)
special=np.minimum(rng.poisson(np.where(seg=='Online TA',.8,.4)),5)
parking=(rng.random(N)<np.where(hotel=='Resort Hotel',.1,.04)).astype(int)
wk=rng.integers(0,3,N)+ (rng.random(N)<.1)*2; wd=rng.integers(0,6,N)+(rng.random(N)<.5)
wd=np.where((wk==0)&(wd==0),1,wd)
adults=rng.choice([1,2,3],N,p=[.2,.75,.05]); children=np.where(rng.random(N)<.08,rng.integers(1,3,N),0)
country=rng.choice(['PRT','GBR','FRA','ESP','DEU','ITA','IRL','BEL','BRA','NLD','USA','CHE','Other'],N,p=[.35,.10,.09,.08,.06,.05,.03,.025,.03,.02,.02,.02,.125])
base=np.where(hotel=='City Hotel',95,80)
season=np.array([0.75,0.78,0.88,1.0,1.08,1.2,1.5,1.6,1.15,0.98,0.8,0.85])[month]
adr=np.clip(base*season*np.where(hotel=='Resort Hotel',1+0.15*(np.isin(month,[6,7,8])),1)+rng.normal(0,18,N)+children*8+(adults-2)*12,0,500).round(2)
# cancellation logit
z=(-2.2+0.0065*lead+np.where(deposit=='Non Refund',3.6,0)+0.55*(prev_c>0)-1.1*repeat-0.55*special-1.2*parking
   +np.where(hotel=='City Hotel',0.35,0)+np.where(seg=='Online TA',0.45,0)+np.where(seg=='Groups',0.3,0)-np.where(seg=='Direct',0.5,0)-np.where(seg=='Corporate',0.7,0))
p=1/(1+np.exp(-z)); canc=(rng.random(N)<p).astype(int)
df=pd.DataFrame(dict(hotel=hotel,is_canceled=canc,lead_time=lead,arrival_date_year=rng.choice([2015,2016,2017],N,p=[.2,.5,.3]),
 arrival_date_month=[month_names[m] for m in month],stays_in_weekend_nights=wk,stays_in_week_nights=wd,adults=adults,children=children,
 country=country,market_segment=seg,is_repeated_guest=repeat,previous_cancellations=prev_c,deposit_type=deposit,adr=adr,
 required_car_parking_spaces=parking,total_of_special_requests=special))
# inject some dirt for cleaning step
dirty=rng.choice(N,600,replace=False); df.loc[dirty[:300],'country']=None
df.loc[dirty[300:420],'adr']=0
df.loc[dirty[420:480],['adults']]=0; df.loc[dirty[420:480],'children']=0
import os
os.makedirs('data',exist_ok=True)
df.to_csv('data/hotel_bookings.csv',index=False)
print('Wrote data/hotel_bookings.csv',df.shape)
