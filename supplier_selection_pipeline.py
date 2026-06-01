import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- BÖLÜM 1: BWM HESAPLAMASI (Kriter Ağırlıkları) ---
print("1. AŞAMA: BWM Ağırlıkları Hesaplanıyor...")

# CSV'den veriyi oku
bwm_df = pd.read_csv('bwm_data.csv', sep=';')
# Excel'deki sayıları alıyoruz
best_to_others = bwm_df.iloc[:, 1].values  # 2. sütun
others_to_worst = bwm_df.iloc[:, 2].values # 3. sütun
n = len(best_to_others)

# Matematiksel optimizasyon fonksiyonu
def bwm_objective(x):
    w = x[:n]
    ksi = x[n]
    # Formül: |w_best - a_best_j * w_j| ve |w_j - a_j_worst * w_worst|
    error1 = np.abs(w[0] - best_to_others * w)
    error2 = np.abs(w - others_to_worst * w[n-1])
    return np.max(np.concatenate([error1, error2])) + ksi

# Kısıtlar: Ağırlık toplamı 1 olmalı
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - 1})
bnds = [(0.0001, 1) for _ in range(n)] + [(0, None)]
initial = [1/n] * n + [0.1]

res = minimize(bwm_objective, initial, bounds=bnds, constraints=cons)
weights = res.x[:n]
ksi_value = res.x[n]

# Sonuçları göster
criteria_names = bwm_df.iloc[:, 0].values
for i, name in enumerate(criteria_names):
    print(f"{name} Ağırlığı: %{weights[i]*100:.2f}")

# --- BÖLÜM 2: GRADIENT BOOSTING (Tedarikçi Analizi) ---
print("\n2. AŞAMA: Makine Öğrenmesi Modeli Eğitiliyor...")

# Tedarikçi verisini oku
supplier_df = pd.read_csv('supplier_data.csv', sep=';')

# X: Kriter puanları, y: Toplam skor
X = supplier_df.iloc[:, 1:6] # C1'den C5'e kadar olan sütunlar
y = supplier_df['TOTAL']

# Veriyi böl (Eğitim ve Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli kur ve eğit
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap ve hata ölç
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Model Eğitimi Tamamlandı. Test Hata Oranı (MSE): {mse:.4f}")
print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
