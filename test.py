from scapy.all import rdpcap, IP, TCP
from collections import defaultdict

def generate_flow_key(packet):
    if packet.haslayer('IP'):
        ip_src = packet['IP'].src
        ip_dst = packet['IP'].dst
        port_src = packet.sport if packet.haslayer('TCP') or packet.haslayer('UDP') else 0
        port_dst = packet.dport if packet.haslayer('TCP') or packet.haslayer('TCP') else 0
        protocol = packet.proto
        return (ip_src, ip_dst, port_src, port_dst, protocol)
    return None

def pcap_to_flows(pcap_file):
    packets = rdpcap(pcap_file)
    flows = defaultdict(lambda: {'timestamps': [], 'sizes': [], 'ttls': [], 'windows': []})
    
    for packet in packets:
        flow_key = generate_flow_key(packet)
        if flow_key:
            timestamp = packet.time
            size = len(packet)
            flows[flow_key]['timestamps'].append(timestamp)
            flows[flow_key]['sizes'].append(size)
            
            if IP in packet:
                ttl = packet[IP].ttl
                flows[flow_key]['ttls'].append(ttl)
                
            if TCP in packet:
                window_size = packet[TCP].window
                flows[flow_key]['windows'].append(window_size)
    
    filtered_flows = {key: value for key, value in flows.items() if len(value['sizes']) >= 2}
    
    return filtered_flows

import numpy as np

def calculate_features(flow_data):
    features_list = []
    for data in flow_data.items():
        feature_values = []
        
        if data['ttls']:
            ttl_array = np.array(data['ttls'])
            feature_values.extend([
                np.mean(ttl_array),
                np.min(ttl_array),
                np.max(ttl_array),
                np.std(ttl_array),
                np.var(ttl_array)
            ])
        else:
            feature_values.extend([None] * 5)

        if data['windows']:
            window_array = np.array(data['windows'])
            feature_values.extend([
                np.mean(window_array),
                np.min(window_array),
                np.max(window_array),
                np.std(window_array),
                np.var(window_array)
            ])
        else:
            feature_values.extend([None] * 5)
        
        features_list.append(feature_values)
    
    return features_list

column_names = [
    'ttl_mean', 'ttl_min', 'ttl_max', 'ttl_std', 'ttl_var',
    'window_mean', 'window_min', 'window_max', 'window_std', 'window_var'
]

# 读取数据并提取流
flows = pcap_to_flows(r"C:\Users\wangl\Desktop\ml_learning\http.pcap")

# 计算特征
features = calculate_features(flows)

for feature in features:
    print(feature)
