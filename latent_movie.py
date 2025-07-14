import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import os

# points: an array of 2 elements, with the first being x value array and the second y val array
def save_curve_frames(latents, points, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    [x_vals, y_vals] = points  # unpack points

    for i in range(0, len(points[0])):
        plt.figure(figsize=(6, 4))

        # Highlight the current point
        x_highlight = x_vals[i]
        y_highlight = y_vals[i]
        plt.plot(latents[0], latents[1], "c.", alpha=0.3)
        plt.plot(points[0], points[1], "g--")
        plt.plot(x_highlight, y_highlight, 'ro', label=f'Point {i+1}')
        plt.grid()

        # Save the frame
        frame_path = os.path.join(output_folder, f"frame_{i+1:03d}.png")
        plt.savefig(frame_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved: {frame_path}")


# loading latent vectors
latent_file = "C:/Users/Siyu_/Desktop/DeepSDF-main/latent_epoch_2000.txt"
latent_data = np.loadtxt(latent_file)

# 2D by PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(latent_data)

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
    tck, _ = splprep(path, s=3)
    u = np.linspace(0, 1, 500)  # interpolation by 500 points
    smooth_path = np.array(splev(u, tck))

    # Inverse conversion 2D -> 40D
    interpolated_latent_vectors = pca.inverse_transform(smooth_path.T)

    # Save Results
    np.savetxt("Locus in latent vector space.txt", interpolated_latent_vectors)

    
    plt.figure
    plt.plot(reduced_data.T[0], reduced_data.T[1], "c.", alpha=0.3)
    plt.plot(smooth_path[0], smooth_path[1], "g--")
    plt.title("Preview")
    plt.show()

    save_curve_frames(reduced_data.T, smooth_path, "C:/Users/Siyu_/Desktop/Model Experiments/latent_animation/frames/graphs")

# # Save locus data (without interpolation)
# if path:
#     path = np.array(path)  # 軌跡をnumpy配列に変換
#     selected_latent_vectors = pca.inverse_transform(path)  # 2D -> 40Dに逆変換
#     np.savetxt("Locus in latent vector space.txt", selected_latent_vectors)