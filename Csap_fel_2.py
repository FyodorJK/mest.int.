from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

okt_df = pd.read_csv("", header = 1, sep=';', encoding='ISO-8859-2')
nep_df = pd.read_csv("", header=1, sep=';', encoding='ISO-8859-2')

for col in okt_df.columns[1:]:
    okt_df[col] = okt_df[col].astype(str).str.replace(" ", "").str.replace(",", ".")
    okt_df[col] = pd.to_numeric(okt_df[col], errors="coerce")

for col in nep_df.columns[1:]:
    nep_df[col] = nep_df[col].astype(str).str.replace(" ", "").str.replace(",", ".")
    nep_df[col] = pd.to_numeric(nep_df[col], errors="coerce")

okt_df_filtered = okt_df[okt_df["Év"] > 2000].dropna()


osszesen_row = nep_df[nep_df.iloc[:, 0].str.contains("Összesen", case=False, na=False)].iloc[0]


columns_from_2000 = [col for col in nep_df.columns if str(col).isdigit() and int(col) > 2000 and int(col) < 2024]


osszesen_from_2000 = osszesen_row[columns_from_2000].astype(int)

okt_df_filtered = okt_df_filtered.iloc[:, :3]

osszesen_df = pd.DataFrame({'Év': osszesen_from_2000.index.astype(int), 'Összesen': osszesen_from_2000.values})



okt_df_filtered["Atlag"] = okt_df_filtered[["Jelentkezett", "Felvett"]].mean(axis=1)

merged_df = pd.merge(okt_df_filtered, osszesen_df[["Év", "Összesen"]], on="Év", how="inner")

merged_df["Jelentkezettek a népességből"] = merged_df["Összesen"] / merged_df["Jelentkezett"]
merged_df["Felvettek a népességből"] = merged_df["Összesen"] / merged_df["Felvett"]



plt.figure(figsize=(10, 6))
plt.plot(merged_df["Év"], merged_df["Atlag"], marker='o', label="Atlag (Felvett, Jelentkezett)")
plt.title("Felvett, Jelentkezett átlaga")
plt.xlabel("Év")
plt.ylabel("Fő")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(merged_df["Év"], merged_df["Jelentkezettek a népességből"], marker='o', label="Jelentkezettek a népességből")
plt.title("Átlag népessége a jelentkezetteknek")
plt.xlabel("Év")
plt.ylabel("Fő")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(merged_df["Év"], merged_df["Felvettek a népességből"], marker='o', label="Felvettek a népesséből")
plt.title("Átlag népessége a felvetteknek")
plt.xlabel("Év")
plt.ylabel("Fő")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

X = merged_df["Év"].values.reshape(-1, 1)
y = merged_df["Összesen"].values


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


poly_model = LinearRegression()
poly_model.fit(X_poly, y)


future_years = np.arange(2024, 2051).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
predictions_poly = poly_model.predict(future_years_poly)

plt.figure(figsize=(10, 6))
plt.plot(merged_df["Év"], merged_df["Összesen"], marker='o', label="Történelmi adat")
plt.plot(future_years, predictions_poly, linestyle='--', color='red', label="Polinomiális predikció")
plt.title("Népesség – Polinomiális Predikció 2050-ig")
plt.xlabel("Év")
plt.ylabel("Népesség (Fő)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

X = merged_df["Év"].values.reshape(-1, 1)
y = merged_df["Jelentkezett"].values
model = LinearRegression().fit(X, y)

future_years = np.arange(2024, 2051).reshape(-1, 1)
predictions = model.predict(future_years)

plt.figure(figsize=(10, 6))
plt.plot(merged_df["Év"], merged_df["Jelentkezett"], marker='o', label="Történelmi adat")
plt.plot(future_years, predictions, linestyle='--', color='red', label="Predikció (ML)")
plt.title("Jelentkezett – Predikció 2050-ig")
plt.xlabel("Év")
plt.ylabel("Fő")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
