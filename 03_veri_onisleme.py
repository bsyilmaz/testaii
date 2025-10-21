import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # Pipeline'ı kaydetmek için

# Veriyi tekrar yükle (ayrı dosya olarak çalıştırılacağı varsayımıyla)
try:
    df = pd.read_csv('Final_data.csv')
except FileNotFoundError:
    print("Final_data.csv bulunamadı. Lütfen dosyayı yükleyin.")
    exit()

print("--- Veri Ön İşleme Başlatılıyor ---")

# 1. Hedef Değişken (y) ve Özellikleri (X) Tanımlama
# Hedefimiz: 'Workout_Type' (Spor Türü) 
# 'Workout_Type' sütununda eksik veri varsa bu satırları eğitimden çıkaralım
df = df.dropna(subset=['Workout_Type'])
y = df['Workout_Type']

# 2. Özellik (Feature) Seçimi
# Modelinize dahil etmek için mantıklı sütunları seçin.
# Çok fazla sütun (53) var, bu nedenle alakasız veya çok fazla eksik veri içerenleri çıkarıyoruz.
# Örnek olarak bu özellikleri seçiyoruz:
numeric_features = [
    'Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM', 
    'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 
    'Experience_Level', 'BMI', 'Calories', 'Carbs', 'Proteins', 'Fats'
] [cite: 53646]

categorical_features = [
    'Gender', 'diet_type'
] [cite: 53646]

# Seçilen özelliklerin veri setinde olduğundan emin olalım
numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

X = df[numeric_features + categorical_features]

print(f"Kullanılan Sayısal Özellikler: {numeric_features}")
print(f"Kullanılan Kategorik Özellikler: {categorical_features}")

# 3. Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
print(f"Test seti boyutu: {X_test.shape[0]} örnek")

# 4. Ön İşleme Pipelaynları Oluşturma
# Sayısal özellikler için: Eksik değerleri medyan ile doldur ve ölçekle
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Kategorik özellikler için: Eksik değerleri en sık görülen ile doldur ve One-Hot-Encoding uygula
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer ile bu iki pipelaynı birleştir
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Seçilmeyen sütunları atar
)

# 5. Ön İşlemciyi (Preprocessor) Kaydetme
# Bu 'preprocessor' nesnesi, 4. ve 5. dosyalarda modellere dahil edilecek.
# Ayrı dosyalarda çalışmak için onu diske kaydedip diğer dosyalarda yükleyebilirsiniz.
joblib.dump(preprocessor, 'colab_preprocessor.joblib')
joblib.dump((X_train, X_test, y_train, y_test), 'colab_train_test_data.joblib')

print("--- Veri Ön İşleme Tamamlandı ---")
print("Preprocessor 'colab_preprocessor.joblib' olarak kaydedildi.")
print("Eğitim/Test verileri 'colab_train_test_data.joblib' olarak kaydedildi.")
