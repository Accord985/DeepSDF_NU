import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev

# loading latent vectors
latent_file = "C:/Users/Siyu_/Desktop/DeepSDF-main/latent_epoch_2000.txt"
latent_data = np.loadtxt(latent_file)

# 2D by PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(latent_data)
print(reduced_data)

# Display 2D plot and hand-draw trajectory
fig, ax = plt.subplots()
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
ax.set_title("Draw a path on the latent space")

path = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        path.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro')  # 点をプロット
        plt.draw()

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()

# Smooth interpolation of trajectory
if len(path) > 2:
    path = np.array(path).T
    tck, _ = splprep(path, s=2)
    u = np.linspace(0, 1, 500)  # interpolation by 500 points
    smooth_path = np.array(splev(u, tck))

    plt.figure
    plt.plot(reduced_data.T[0], reduced_data.T[1], "c.", alpha=0.3)
    plt.plot(smooth_path[0], smooth_path[1], "g--")
    plt.show()

    # Inverse conversion 2D -> 40D
    interpolated_latent_vectors = pca.inverse_transform(smooth_path.T)

    # Save Results
    np.savetxt("Locus in latent vector space.txt", interpolated_latent_vectors)

# # Save locus data (without interpolation)
# if path:
#     path = np.array(path)  # 軌跡をnumpy配列に変換
#     selected_latent_vectors = pca.inverse_transform(path)  # 2D -> 40Dに逆変換
#     np.savetxt("Locus in latent vector space.txt", selected_latent_vectors)