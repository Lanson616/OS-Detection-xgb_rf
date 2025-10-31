import numpy as np
import scipy
import pickle

with open('flows.pkl', 'rb') as file:
    flows = pickle.load(file)

print (len(flows))

# flow2features
def calculate_features(flow_data):
    timestamps = np.array(flow_data['timestamps'], dtype=float)
    sizes = np.array(flow_data['sizes'], dtype=float)
    
    # Calculate duration
    if len(timestamps) < 2:
        duration = 0
    else:
        duration = float(timestamps[-1] - timestamps[0])
    
    # Avoid division by zero
    if duration == 0:
        packets_per_second = 0
        bytes_per_second = 0
    else:
        packets_per_second = float(len(sizes) / duration)
        bytes_per_second = float(sizes.sum() / duration)
    
    # Calculate size stats 
    mean_size = float(sizes.mean())
    stddev_size = float(sizes.std())
    percentiles = np.percentile(sizes, [25, 50, 75])
    min_size = float(sizes.min())
    max_size = float(sizes.max())

    # Calculate IATs
    iats = np.diff(timestamps) 

    if len(iats) == 0:
        mean_iat = std_dev_iat = min_iat = max_iat = iat_variance = 0
    else:
        mean_iat = float(np.mean(iats))
        std_dev_iat = float(np.std(iats))
        min_iat = float(np.min(iats))
        max_iat = float(np.max(iats))
        iat_variance = float(np.var(iats))
    
    # Calculate Entropy
    entropy = scipy.stats.entropy(sizes)

    # TTL
    if flow_data['ttls']:
        ttl_array = np.array(flow_data['ttls'])

        mean_ttl = np.mean(ttl_array)
        std_dev_ttl = np.std(ttl_array)
        max_ttl = np.max(ttl_array)
        min_ttl = np.min(ttl_array)
        ttl_variance = np.var(ttl_array)
    else:
        mean_ttl = std_dev_ttl = max_ttl = min_ttl = ttl_variance = 0

    # Windows
    if flow_data['windows']:
        window_array = np.array(flow_data['windows'])

        mean_window = np.mean(window_array)
        std_dev_window = np.std(window_array)
        max_window = np.max(window_array)
        min_window = np.min(window_array)
        window_variance = np.var(window_array)
    else:
        mean_window = std_dev_window = max_window = min_window = window_variance = 0

    return [
        duration,
        packets_per_second,
        bytes_per_second,
        mean_size,
        stddev_size,
        percentiles[0],
        percentiles[1],
        percentiles[2],
        min_size,
        max_size,
        len(sizes),
        float(sizes.sum()),
        mean_iat,
        std_dev_iat,
        min_iat,
        max_iat,
        iat_variance,
        entropy,
        mean_ttl,
        std_dev_ttl,
        max_ttl,
        min_ttl,
        ttl_variance,
        mean_window,
        std_dev_window,
        max_window,
        min_window,
        window_variance
        ]

labels = ['duration', 'packets per second', 
          'bytes per second', 'mean',
          'standard deviation', '25', 
          'median', '75', 'minimum', 
          'maximum', 'packets', 'bytes',
          'mean iat', 'std_dev iat',
          'min iat', 'max iat', 
          'iat variance','entropy', 
          'mean_ttl', 'std_dev_ttl',
          'max_ttl', 'min_ttl',
          'ttl_variance', 'mean_window',
          'std_dev_window', 'max_window',
          'min_window', 'window_variance'
          ]

# Extract features
feature_vectors = []

for index, (flow_key, flow_data) in enumerate(flows.items(), start=1):
    features = calculate_features(flow_data)
    feature_vectors.append(features)
    #print(f'Flow {index}: {features}')

X = np.array(feature_vectors)

with open('feature_array.txt', 'w') as file:
    for feature_vector in feature_vectors:
        file.write(','.join(map(str, feature_vector)) + '\n')

print(X)
print("Feature matrix shape:", X.shape)
