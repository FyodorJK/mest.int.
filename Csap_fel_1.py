from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


arak_df = pd.read_csv("", sep=";", encoding="ISO-8859-2", header=1)


for col in arak_df.columns[2:]:
    arak_df[col] = arak_df[col].astype(str).str.replace(" ", "").str.replace(",", ".")
    arak_df[col] = pd.to_numeric(arak_df[col], errors="coerce")


kenyer = arak_df[arak_df["Termék, szolgáltatás"].str.contains("kenyér", case=False, na=False)].iloc[0]
tej = arak_df[arak_df["Termék, szolgáltatás"].str.contains("tej, 2,8", case=False, na=False)].iloc[0]
csirke = arak_df[arak_df["Termék, szolgáltatás"].str.contains("csirke", case=False, na=False)].iloc[0]


arak_df_selected = pd.DataFrame({
    "Év": arak_df.columns[2:].astype(int),
    "Kenyér": kenyer[2:].values,
    "Tej": tej[2:].values,
    "Hús": csirke[2:].values
})


arak_df_selected["Átlagár"] = arak_df_selected[["Kenyér", "Tej", "Hús"]].mean(axis=1)

      
df_jov = pd.read_csv("", header=0)
df_jov.columns = [
    "Év",
    "Bruttó átlagkereset",
    "Nettó átlagkereset",
    "Bruttó index",
    "Nettó index",
    "Forrás"
]
df_jov["Év"] = df_jov["Év"].astype(int)
df_jov = df_jov.sort_values("Év")

      
df_merged = pd.merge(arak_df_selected, df_jov[["Év", "Nettó átlagkereset"]], on="Év", how="inner")


df_merged["Kenyeret vehet (db)"] = df_merged["Nettó átlagkereset"] / df_merged["Kenyér"]
df_merged["Tejet vehet (l)"] = df_merged["Nettó átlagkereset"] / df_merged["Tej"]
df_merged["Húst vehet (kg)"] = df_merged["Nettó átlagkereset"] / df_merged["Hús"]




plt.figure(figsize=(10, 6))
plt.plot(df_merged["Év"], df_merged["Átlagár"], marker='o', label="Átlagár (kenyér, tej, hús)")
plt.title("Kenyér, tej, hús átlagára")
plt.xlabel("Év")
plt.ylabel("Ft")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df_merged["Év"], df_merged["Kenyeret vehet (db)"], marker='o', label="Kenyeret vehet (db)")
plt.title("Átlagbérből vásárolható kenyér mennyisége")
plt.xlabel("Év")
plt.ylabel("Darab")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_merged["Év"], df_merged["Tejet vehet (l)"], marker='o', label="Tejet vehet (liter)")
plt.title("Átlagbérből vásárolható tej mennyisége")
plt.xlabel("Év")
plt.ylabel("Liter")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_merged["Év"], df_merged["Húst vehet (kg)"], marker='o', label="Húst vehet (kg)")
plt.title("Átlagbérből vásárolható hús mennyisége")
plt.xlabel("Év")
plt.ylabel("Kilogramm")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



X = df_merged["Év"].values.reshape(-1, 1)
y = df_merged["Nettó átlagkereset"].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)


future_years = np.arange(2024, 2051).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
predictions_poly = poly_model.predict(future_years_poly)


plt.figure(figsize=(10, 6))
plt.plot(df_merged["Év"], df_merged["Nettó átlagkereset"], marker='o', label="Történelmi adat")
plt.plot(future_years, predictions_poly, linestyle='--', color='red', label="Polinomiális predikció")
plt.title("Nettó átlagbér – Polinomiális Predikció 2050-ig")
plt.xlabel("Év")
plt.ylabel("Nettó átlagbér (Ft)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



X = df_merged["Év"].values.reshape(-1, 1)
y = df_merged["Kenyeret vehet (db)"].values
model = LinearRegression().fit(X, y)

future_years = np.arange(2024, 2051).reshape(-1, 1)
predictions = model.predict(future_years)

plt.figure(figsize=(10, 6))
plt.plot(df_merged["Év"], df_merged["Kenyeret vehet (db)"], marker='o', label="Történelmi adat")
plt.plot(future_years, predictions, linestyle='--', color='red', label="Predikció (ML)")
plt.title("Kenyeret vehet (db) – Predikció 2050-ig")
plt.xlabel("Év")
plt.ylabel("Darab")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
