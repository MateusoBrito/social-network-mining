import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.classification.classic_pipeline import run_classic_pipeline
from src.utils.data_loader import load_all_data, ROOT


COLUMNS = [
    'titleSize',
    'titleChars',
    'tem_hashtag',
    'tem_url',
    'commentsCount',
    'hora_num',
    'periodo_dia',
    'engajamento',
    'lang',
    'grau',
    'centralidade_autovetor',
    'coeficiente_clusterizacao',
]

df = load_all_data()
df = df[COLUMNS].copy()

# Detecta colunas categóricas
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

# Salva encoders caso queira interpretar depois
encoders = {}

# Converte categóricas para números
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],         
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None] 
}

FEATURE_COLUMNS = [col for col in df.columns if col != 'engajamento']

X = df[FEATURE_COLUMNS].values
y = df['engajamento'].values

run_classic_pipeline(
    X = X,
    y_encoded= y,
    splitter=skf,
    model_class=RandomForestClassifier,
    param_grid=grid,
    model_name="RandomForest_Structural_KDD",
    dataset_path="reddit_processed.csv",
    output_dir="./results",
    class_names=["Low Engagement", "High Engagement"]
)
