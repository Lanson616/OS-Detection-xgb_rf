from collections import defaultdict

from scapy.all import rdpcap, IP, TCP

def generate_flow_key(packet):
    if packet.haslayer('IP'):
        ip_src = packet['IP'].src
        ip_dst = packet['IP'].dst
        port_src = packet.sport if packet.haslayer('TCP') or packet.haslayer('UDP') else 0
        port_dst = packet.dport if packet.haslayer('TCP') or packet.haslayer('TCP') else 0
        protocol = packet.proto
        return (ip_src, ip_dst, port_src, port_dst, protocol)
    return None

# pcap2flows
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


flows = pcap_to_flows(r"C:\Users\wangl\Desktop\os_detection_project\os-100-packet.pcapng")

print (len(flows))

import pickle

with open('flows.pkl', 'wb') as file:
    pickle.dump(flows, file)

with open('flows.pkl', 'rb') as file:
    loaded_flows = pickle.load(file)

print(loaded_flows)
