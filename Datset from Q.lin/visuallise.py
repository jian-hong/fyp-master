import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Load the .mat file
# Make sure 'data1.0mm.mat' is in the same folder as this script
mat_data = sio.loadmat('data1.0mm.mat')

# Print keys to confirm what is actually inside
print("Keys found:", mat_data.keys())

# --- THE FIX ---
# Your file does NOT have 'data_x'. It has 'data10mm'.
# We extract that specific key.
y = mat_data['data10mm'].flatten() 

# We create an artificial X-axis (sample count) because the file didn't provide one
x = np.arange(len(y))

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Partial Discharge Signal (data10mm)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()