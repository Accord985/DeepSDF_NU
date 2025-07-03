# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import os

# # ディレクトリとファイルのパス
# directory = "D:/porosity_data/鋳巣データ/形状学習/3回目"
# latent_file = os.path.join(directory, "latent.txt")
# volume_file = os.path.join(directory, "体積.txt")

# # データの読み込み
# latent_data = np.loadtxt(latent_file)

# # 体積データの読み込み
# volume_data = []
# with open(volume_file, "r", encoding="utf-8") as f:
#     for line in f:
#         try:
#             volume = float(line.strip())
#             volume_data.append(volume)
#         except ValueError:
#             continue

# # print(len(volume_data), len(latent_data))

# # 体積データの個数が latent データと一致するか確認
# if len(volume_data) != len(latent_data):
#    raise ValueError("体積データと潜在ベクトルのデータ数が一致しません！")

# # PCAで次元削減
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(latent_data)

# # 結果を pandas の DataFrame に変換
# df = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
# df["Volume"] = volume_data  # 体積データを追加

# # Excelファイルに保存
# output_file = os.path.join(directory, "PCA_results.xlsx")
# df.to_excel(output_file, index=False)

# print(f"PCAの結果を {output_file} に保存しました。")

# # 可視化
# plt.figure(figsize=(8, 6))
# sc = plt.scatter(df["PC1"], df["PC2"], c=df["Volume"], cmap="viridis", alpha=0.7, vmin=0, vmax=0.08)
# plt.colorbar(sc, label="Volume (mm³)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA Visualization of Latent Vectors (Colored by Volume)")
# plt.show()





# Visualizes the latent space with latent vector logs. 

# 体積不要版
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ディレクトリとファイルのパス
directory = "C:/Users/Siyu_/Desktop/DeepSDF-main"
latent_file = os.path.join(directory, "latent_epoch_2000.txt")

# データの読み込み
latent_data = np.loadtxt(latent_file)

# PCAで次元削減
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(latent_data)
   
# 結果を pandas の DataFrame に変換
df = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])

# Excelファイルに保存
output_file = os.path.join(directory, "PCA.xlsx")
df.to_excel(output_file, index=False)

print(f"PCAの結果を {output_file} に保存しました。")

# 可視化
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["PC1"], df["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization of Latent Vectors (Colored by Volume)")
plt.show()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import os

# # ディレクトリとファイルのパス
# directory = "D:/porosity_data/鋳巣データ/形状学習/3回目"
# latent_file = os.path.join(directory, "latent.txt")
# volume_file = os.path.join(directory, "体積.txt")

# # データの読み込み
# latent_data = np.loadtxt(latent_file)

# # 体積データの読み込み
# volume_data = []
# with open(volume_file, "r", encoding="utf-8") as f:
#     for line in f:
#         try:
#             volume = float(line.strip())
#             volume_data.append(volume)
#         except ValueError:
#             continue

# # print(len(volume_data), len(latent_data))

# # 体積データの個数が latent データと一致するか確認
# if len(volume_data) != len(latent_data):
#    raise ValueError("体積データと潜在ベクトルのデータ数が一致しません！")

# # PCAで次元削減
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(latent_data)

# # 結果を pandas の DataFrame に変換
# df = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
# df["Volume"] = volume_data  # 体積データを追加

# # Excelファイルに保存
# output_file = os.path.join(directory, "PCA_results.xlsx")
# df.to_excel(output_file, index=False)

# print(f"PCAの結果を {output_file} に保存しました。")

# # 可視化
# plt.figure(figsize=(8, 6))
# sc = plt.scatter(df["PC1"], df["PC2"], c=df["Volume"], cmap="viridis", alpha=0.7, vmin=0, vmax=0.08)
# plt.colorbar(sc, label="Volume (mm³)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA Visualization of Latent Vectors (Colored by Volume)")
# plt.show()





# Visualizes the latent space with latent vector logs. 

# 体積不要版
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ディレクトリとファイルのパス
directory = "C:/Users/Siyu_/Desktop/DeepSDF-main"
latent_file = os.path.join(directory, "latent_epoch_2000.txt")

# データの読み込み
latent_data = np.loadtxt(latent_file)

# PCAで次元削減
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(latent_data)
   
# 結果を pandas の DataFrame に変換
df = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])

# Excelファイルに保存
output_file = os.path.join(directory, "PCA.xlsx")
df.to_excel(output_file, index=False)

print(f"PCAの結果を {output_file} に保存しました。")

# 可視化
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["PC1"], df["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization of Latent Vectors (Colored by Volume)")
plt.show()