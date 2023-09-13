import numpy as np
import pickle
import pandas as pd
import os


model_features_RGB_train=pd.DataFrame(pd.read_pickle('saved_features/saved_feat_I3D_S04_train.pkl')['features'])[["uid", "features_" + 'RGB']]
model_features_EMG_train=pd.DataFrame(pd.read_pickle('saved_features/saved_feat_EMG_S04_train.pkl')['features'])[["uid", "features_" + 'RGB']]
model_features_RGB_test=pd.DataFrame(pd.read_pickle('saved_features/saved_feat_I3D_S04_test.pkl')['features'])[["uid", "features_" + 'RGB']]
model_features_EMG_test=pd.DataFrame(pd.read_pickle('saved_features/saved_feat_EMG_S04_test.pkl')['features'])[["uid", "features_" + 'RGB']]

tmp_uid=[]
tmp_concat=[]
tmp_sum=[]
tmp_avg=[]
for i in model_features_RGB_train['uid']:
    if i!=2863:
        a=model_features_RGB_train.query(f"uid=={i}")['features_RGB'].iloc[0]
        b=model_features_EMG_train.query(f"uid=={i}")['features_RGB'].iloc[0]
        tmp_sum.append(a+b)#sum
        d=[(a.mean(axis=0)+b[0])/2] # average, b is equal for all clips
        d=np.array([d,d,d,d,d])#creating 5 vectors to simulate clips   
        tmp_avg.append(d)
        tmp_concat.append(np.concatenate((a,b),axis=1))
        tmp_uid.append(i)

list_train_concat=[{'uid': i,'video_name':"", 'features_RGB': j} for i,j in zip(tmp_uid,tmp_concat)]
mixed_feat_train_concat={'features':list_train_concat}
list_train_sum=[{'uid': i,'video_name':"", 'features_RGB': j} for i,j in zip(tmp_uid,tmp_sum)]
mixed_feat_train_sum={'features':list_train_sum}
list_train_avg=[{'uid': i,'video_name':"", 'features_RGB': j} for i,j in zip(tmp_uid,tmp_avg)]
mixed_feat_train_avg={'features':list_train_avg}

tmp_concat=[]
tmp_uid=[]
tmp_sum=[]
tmp_avg=[]
for i in model_features_RGB_test['uid']:
    if i!=2778:
        a=model_features_RGB_test.query(f"uid=={i}")['features_RGB'].iloc[0]
        b=model_features_EMG_test.query(f"uid=={i}")['features_RGB'].iloc[0]
        tmp_sum.append(a+b)#sum
        d=(a.mean(axis=0)+b[0])/2 # average, b is equal for all clips
        d=np.array([d,d,d,d,d])#creating 5 vectors to simulate clips np.array([f,f,f,f,f])  
        tmp_avg.append(d)
        tmp_concat.append(np.concatenate((a,b),axis=1))
        tmp_uid.append(i)


list_test_concat=[{'uid': i,'video_name':"", 'features_RGB': j} for i,j in zip(tmp_uid,tmp_concat)]
mixed_feat_test_concat={'features':list_test_concat}
list_test_sum=[{'uid': i,'video_name':"", 'features_RGB': j} for i,j in zip(tmp_uid,tmp_sum)]
mixed_feat_test_sum={'features':list_test_sum}
list_test_avg=[{'uid': i,'video_name':"", 'features_RGB': j} for i,j in zip(tmp_uid,tmp_avg)]
mixed_feat_test_avg={'features':list_test_avg}

# pickle.dump(mixed_feat_test_concat,open('./saved_features/saved_feat_MIXED_CONCAT_S04_test.pkl','wb'))
# pickle.dump(mixed_feat_train_concat,open('./saved_features/saved_feat_MIXED_CONCAT_S04_train.pkl','wb'))

# pickle.dump(mixed_feat_test_sum,open('./saved_features/saved_feat_MIXED_SUM_S04_test.pkl','wb'))
# pickle.dump(mixed_feat_train_sum,open('./saved_features/saved_feat_MIXED_SUM_S04_train.pkl','wb'))

pickle.dump(mixed_feat_test_avg,open('./saved_features/saved_feat_MIXED_AVG_S04_test.pkl','wb'))
pickle.dump(mixed_feat_train_avg,open('./saved_features/saved_feat_MIXED_AVG_S04_train.pkl','wb'))