import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bir önceki dosyadan 'df' değişkeninin yüklendiğini varsayıyoruz.
# Eğer bu dosyayı ayrı çalıştıracaksanız, veri yükleme kodunu buraya tekrar eklemelisiniz.
# df = pd.read_csv('Final_data.csv')

def run_eda(dataframe):
    """
    Temel EDA görselleştirmelerini yapar.
    """
    if dataframe is None:
        print("Veri seti yüklenemedi. EDA çalıştırılamıyor.")
        return

    print("\n--- Keşifçi Veri Analizi (EDA) Başlatılıyor ---")
    
    # Hedef değişkenimizin (Workout_Type) dağılımı
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataframe, x='Workout_Type')
    plt.title('Spor Türü (Workout_Type) Dağılımı')
    plt.xlabel('Spor Türü')
    plt.ylabel('Sayı')
    plt.savefig('01_workout_type_dagilimi.png')
    print("Görsel '01_workout_type_dagilimi.png' olarak kaydedildi.")
    plt.close()

    # Önemli sayısal değişkenlerin dağılımı (Histogram)
    numeric_features_to_plot = ['Age', 'Weight (kg)', 'Calories_Burned', 'Session_Duration (hours)']
    for col in numeric_features_to_plot:
        if col in dataframe.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(dataframe[col], kde=True, bins=30)
            plt.title(f'{col} Dağılımı')
            plt.savefig(f'02_{col}_histogram.png')
            print(f"Görsel '02_{col}_histogram.png' olarak kaydedildi.")
            plt.close()

    # Cinsiyete göre kalori yakımı (Boxplot)
    if 'Gender' in dataframe.columns and 'Calories_Burned' in dataframe.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=dataframe, x='Gender', y='Calories_Burned')
        plt.title('Cinsiyete Göre Yakılan Kalori')
        plt.savefig('03_cinsiyet_kalori_boxplot.png')
        print("Görsel '03_cinsiyet_kalori_boxplot.png' olarak kaydedildi.")
        plt.close()

    # Sayısal sütunlar arasındaki korelasyon matrisi
    # Yalnızca modelde kullanılması muhtemel sütunları seçiyoruz
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    relevant_numeric_cols = [
        'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
        'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
        'Water_Intake (liters)', 'Workout_Frequency (days/week)', 
        'Experience_Level', 'BMI', 'Calories'
    ]
    # Sadece var olan sütunları al
    relevant_numeric_cols = [col for col in relevant_numeric_cols if col in numeric_cols]
    
    if relevant_numeric_cols:
        corr_matrix = dataframe[relevant_numeric_cols].corr()
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
        plt.title('Sayısal Değişkenler Korelasyon Matrisi')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('04_korelasyon_heatmap.png')
        print("Görsel '04_korelasyon_heatmap.png' olarak kaydedildi.")
        plt.close()
    
    print("--- EDA Tamamlandı ---")

# Bu fonksiyonu çalıştırmak için ana veri setini (df) kullanın
# Önceki dosyada 'df' yüklendiği için burada tekrar yüklüyoruz (ayrı dosya mantığı için)
try:
    df_eda = pd.read_csv(CSV_FILE_PATH)
    run_eda(df_eda)
except NameError:
    print("Lütfen 'df' değişkeninin yüklendiği 01 numaralı dosyayı önce çalıştırın.")
except Exception as e:
    print(f"EDA sırasında bir hata oluştu: {e}")
