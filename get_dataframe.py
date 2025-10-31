import pandas as pd
from scapy.all import rdpcap, IP, TCP
from collections import defaultdict

def generate_flow_key(packet):
    if packet.haslayer('IP'):
        ip_src = packet['IP'].src
        ip_dst = packet['IP'].dst
        port_src = packet.sport if packet.haslayer('TCP') or packet.haslayer('UDP') else 0
        port_dst = packet.dport if packet.haslayer('TCP') or packet.haslayer('UDP') else 0
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

def flows_to_dataframe(flows):
    rows = []
    for key, value in flows.items():
        ip_src, ip_dst, port_src, port_dst, protocol = key
        row = {
            'ip_src': ip_src,
            'ip_dst': ip_dst,
            'port_src': port_src,
            'port_dst': port_dst,
            'protocol': protocol,
            'timestamps': value['timestamps'],
            'sizes': value['sizes'],
            'ttls': value['ttls'],
            'windows': value['windows']
        }
        rows.append(row)
    return pd.DataFrame(rows)

flows = pcap_to_flows(r"C:\Users\wangl\Desktop\os_detection_project\os-100-packet.pcapng")
df = flows_to_dataframe(flows)

df.to_csv('dataframe.csv', index=False)

