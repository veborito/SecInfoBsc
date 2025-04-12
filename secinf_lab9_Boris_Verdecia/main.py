import cmd
from scapy.all import rdpcap
from collections import Counter

class MyCLI(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.file_loaded = False

    def do_load(self, filepath):
        """Load a pcap file"""
        self.packets = rdpcap(filepath)
        self.file_loaded = True

    def do_list_ip_addresses(self, line):
        """List all the IP addresses viewed in the pcap file"""
        if self.file_loaded:
            list_ip_addresses(self.packets)
        else:
            print("You must first load a file using: load <filepath>")

    def do_overview(self):
        """Get an overview of the pcap file"""
        if self.file_loaded:
            overview(self.packets)
        else:
            print("You must first load a file using: load <filepath>")

    def do_stats(self, ip_address):
        """Get stats from the point of view of a specific host"""
        if self.file_loaded:
            stats(self.packets, ip_address)
        else:
            print("You must first load a file using: load <filepath>")

    def do_quit(self, line):
        return True

class peer():
    def __init__(self):
        self.sent_bytes = 0
        self.received_bytes = 0
        self.sent_packets = 0
        self.received_packets = 0


def list_ip_addresses(packets):
    # Initilisation d'un set
    all_ip_addresses = set()
    for p in packets:
        if p.haslayer('IP'):
            # on ajoute au set les nouvelles adresses ip non existantes
            all_ip_addresses.add(p['IP'].dst)
            all_ip_addresses.add(p['IP'].src)
    # on affiche toutes les adresses ip triées            
    for ip in sorted(all_ip_addresses):
        print(ip)
        
def overview(packets):
     # Initialisation d'un dictionnaire avec toutes les adresses ip uniques
    # associées à un objet peer
    peers = dict()
    for p in packets:
        if p.haslayer('IP'):
            peers[p['IP'].src] = peer()
            peers[p['IP'].dst] = peer()
              
    tot_packets = 0
    tot_size = 0
    syn_ack_count = 0
    syn_req = [] # liste des requêtes syn envoyées
    for p in packets:
        if p.haslayer('TCP'):
            if p['TCP'].flags == "S":
                # on récupère les informations de la requête syn afin de pouvoir retrouver 
                # la requête syn-ack correspondante plus tard
                syn_info = (p['IP'].src, p['IP'].dst, p['TCP'].sport, p['TCP'].dport)
                syn_req.append(syn_info)
            if p['TCP'].flags == "SA":
                # On met les informations dans cet ordre
                # afin de pouvoir trouver une correspondance dans la liste de requêtes syn
                syn_ack_info = (p['IP'].dst, p['IP'].src, p['TCP'].dport, p['TCP'].sport) 
                if syn_ack_info not in syn_req: # on check la correspondance avec une précédente requête syn
                    syn_ack_count += 1 # si la requête syn-ack n'a pas de requête correspondante elle est illégale.
        if p.haslayer('IP'):
            # On ajoute toutes les valeurs nécessaires aux objets peer
            # associés aux adresses ip source et destination pour l'overview
            tot_packets += 1
            tot_size += p['IP'].len
            peers[p['IP'].src].sent_bytes += p['IP'].len
            peers[p['IP'].src].sent_packets += 1
            peers[p['IP'].dst].received_bytes += p['IP'].len
            peers[p['IP'].dst].received_packets += 1
    
  
    # on affiche les informations
    print(f"Total packets: {tot_packets}\nTotal size: {tot_size}\nUnique IP addresses: {len(peers.keys())}")
    print(f"Illegal SYN-ACK: {syn_ack_count}")
    print("Peers")
    for k, v in peers.items():
        print(f"    -{k} : {v.sent_bytes} (sent bytes), {v.sent_packets} (sent packets), {v.received_bytes} (received bytes), {v.received_packets} (received packets)")
        
    # DEBUG        
    # print(f"LOG(verification): tot sent bytes sum = {sum([v.sent_bytes for k,v in peers.items()])}, total size = {tot_size}")
    # print(f"LOG(verification): tot received bytes sum = {sum([v.received_bytes for k,v in peers.items()])}, total size = {tot_size}")
    # print(f"LOG(verification): tot sent packets sum = {sum([v.sent_packets for k,v in peers.items()])}, total packets = {tot_packets}")
    # print(f"LOG(verification): tot received packets sum = {sum([v.received_packets for k,v in peers.items()])}, total packets = {tot_packets}")
    
            
def stats(packets, ip_address):
    peers = dict()
    for p in packets:
        if p.haslayer('IP'):
            if p['IP'].src == ip_address or p['IP'].dst == ip_address:
                peers[p['IP'].src] = peer()
                peers[p['IP'].dst] = peer()

    for p in packets:
        if p.haslayer('IP'):
            # on fait pareil que pour l'overview sauf qu'on trie pour les valeurs de l'adresse ip cible
            if p['IP'].src == ip_address or p['IP'].dst == ip_address:
                peers[p['IP'].src].sent_bytes += p['IP'].len
                peers[p['IP'].src].sent_packets += 1
                peers[p['IP'].dst].received_bytes += p['IP'].len
                peers[p['IP'].dst].received_packets += 1
    target_ip = peers.pop(ip_address) # on prend l'information globale sur l'adresse ip cible
    print(f"Total sent: {target_ip.sent_bytes} (bytes), {target_ip.sent_packets} (packets)\n" +
          f"Total received: {target_ip.received_bytes} (bytes), {target_ip.received_packets} (packets)")
    print("Peers")
    for k, v in peers.items():
        print(f"    -{k} : {v.sent_bytes} (sent bytes), {v.sent_packets} (sent packets), {v.received_bytes} (received bytes), {v.received_packets} (received packets)")
    # DEBUG        
    # print(f"LOG(verification): tot sent bytes sum = {sum([v.sent_bytes for k,v in peers.items()])}")
    # print(f"LOG(verification): tot received bytes sum = {sum([v.received_bytes for k,v in peers.items()])}")
    # print(f"LOG(verification): tot sent packets sum = {sum([v.sent_packets for k,v in peers.items()])}")
    # print(f"LOG(verification): tot received packets sum = {sum([v.received_packets for k,v in peers.items()])}")


if __name__ == "__main__":
    MyCLI().cmdloop()
