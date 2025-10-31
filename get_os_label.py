import pyshark
from scapy.all import rdpcap
from collections import defaultdict
import numpy as np
import pickle

with open('flows.pkl', 'rb') as file:
    flows = pickle.load(file)

# Get first_packet_index
first_packet_indices = []

for flow_key, flow_data in flows.items():
    first_packet_index = min(flow_data['indices'])
    first_packet_indices.append(first_packet_index)

print(len(first_packet_indices))

# Get first_packet_comments
with open('os_system_data.txt', 'r') as file:
    all_os = [line.strip() for line in file]

print(len(all_os))

selected_os = [all_os[i] for i in first_packet_indices]

print(len(selected_os))

with open('os_system_label.txt', 'w') as file:
    for os_system in selected_os:
        file.write(f"{os_system}\n")