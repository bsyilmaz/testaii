import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Google Colab ortamına yüklediğiniz CSV dosyasının adını buraya yazın
CSV_FILE_PATH = 'Final_data.csv'

# Veri setini yükle
try:
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Veri setinin ilk birkaç satırını göster
    print("--- Veri Seti İlk 5 Satır ---")
    print(df.head())
    
    # Veri seti hakkında genel bilgileri (sütun tipleri, boş değerler) göster
    print("\n--- Veri Seti Bilgileri ---")
    df.info()
    
    # Sayısal sütunlar için istatistiksel özet
    print("\n--- Sayısal Veri Özeti ---")
    print(df.describe())
    
    # Kategorik sütunlar için istatistiksel özet
    print("\n--- Kategorik Veri Özeti ---")
    print(df.describe(include=['object']))
    
    # Eksik değerlerin toplamını göster
    print("\n--- Eksik Değer Sayıları ---")
    print(df.isnull().sum())

except FileNotFoundError:
    print(f"HATA: '{CSV_FILE_PATH}' dosyası bulunamadı.")
    print("Lütfen dosyayı Google Colab çalışma dizininize yüklediğinizden emin olun.")
