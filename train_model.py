"""
Script d'entraînement du modèle NSL-KDD.
Lance: python train_model.py
"""
import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

DATASET_PATH = "KDDTrain+.txt"

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'
]

FEATURES_15 = [
    'duration','protocol_type','service','flag','src_bytes',
    'dst_bytes','wrong_fragment','hot','logged_in','num_compromised',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate'
]

CATEGORICALS = ['protocol_type', 'service', 'flag']

print("=" * 55)
print("  NSL-KDD Model Trainer")
print("=" * 55)

if not os.path.exists(DATASET_PATH):
    print(f"\n❌ '{DATASET_PATH}' introuvable.")
    print("📥 Télécharge sur : https://www.kaggle.com/datasets/hassan06/nslkdd")
    sys.exit(1)

print("\n📂 Chargement...")
df = pd.read_csv(DATASET_PATH, header=None)
df.columns = COLUMNS
print(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Binariser la cible : normal=0, attack=1
df['attack'] = (df['attack'] != 'normal').astype(int)
print(f"   Normal: {(df['attack']==0).sum()} | Attack: {(df['attack']==1).sum()}")

# Encodeurs SÉPARÉS pour chaque variable catégorielle
print("\n🔤 Encodage catégoriel...")
label_encoders = {}
for col in CATEGORICALS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   {col}: {len(le.classes_)} classes")

X = df[FEATURES_15].copy()
y = df['attack']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=43, stratify=y
)

print(f"\n✂️  Train: {len(X_train)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\n🤖 Entraînement XGBoost...")
model = XGBClassifier(
    colsample_bytree=0.5, learning_rate=0.1,
    max_depth=6, n_estimators=128,
    subsample=0.8, random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=['normal','attack']))

# Vérifier l'ordre des classes
print(f"\n🔍 Classes du modèle: {model.classes_}")
print(f"   → proba[:,0] = normal | proba[:,1] = attack")

artifacts = {
    "model": model,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "features": FEATURES_15,
    "class_order": list(model.classes_),  # [0=normal, 1=attack]
    "version": "1.0.0",
    "accuracy": round(acc, 4),
}

joblib.dump(artifacts, "model.pkl")
size = os.path.getsize("model.pkl") / 1024 / 1024
print(f"\n✅ model.pkl sauvegardé ({size:.1f} MB)")
print("\n🚀 Lance l'API: uvicorn main:app --reload --port 8000")
