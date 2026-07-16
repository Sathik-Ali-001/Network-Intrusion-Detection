import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('intrusion_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')
le_y = joblib.load('label_encoder.pkl')

feature_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate']

def predict_intrusion(input_data):
    df = pd.DataFrame([input_data])
    df.columns = feature_names
    
    for col in ['protocol_type', 'service', 'flag']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    numeric_cols = [col for col in feature_names if col not in ['protocol_type', 'service', 'flag']]
    X_num = df[numeric_cols].select_dtypes(include=[np.number])
    
    for col in ['protocol_type', 'service', 'flag']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    X = df[feature_names]
    X = X.select_dtypes(include=[np.number])
    
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    
    pred_encoded = model.predict(X_selected)
    pred_label = le_y.inverse_transform(pred_encoded)
    
    return pred_label[0]

if __name__ == "__main__":
    example_input = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 181, 'dst_bytes': 5450, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
        'is_guest_login': 0, 'count': 8, 'srv_count': 8, 'serror_rate': 0.0,
        'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 9, 'dst_host_srv_count': 9, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.11,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    result = predict_intrusion(example_input)
    print("Predicted Attack Type:", result)