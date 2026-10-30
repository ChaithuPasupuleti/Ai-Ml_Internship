import numpy as np

arr_1d = np.array([1, 2, 3, 4])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("1D Shape:", arr_1d.shape)
print("2D Shape:", arr_2d.shape)
print("3D Shape:", arr_3d.shape)

scaled_array = arr_2d * 10
matrix_mult = arr_2d @ arr_2d.T
print("Scaled Array:\n", scaled_array)
print("Matrix Multiplication:\n", matrix_mult)

data = np.array([[25, 50000], [30, 60000], [35, 80000], [40, 110000]])
mean_vals = np.mean(data, axis=0)
std_vals = np.std(data, axis=0)
correlation_matrix = np.corrcoef(data, rowvar=False)

print("Means:", mean_vals)
print("Standard Deviations:", std_vals)
print("Correlation Matrix:\n", correlation_matrix)
#day 1 completed