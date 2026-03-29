import networkx as nx
import matplotlib.pyplot as plt

def render_erg_diagram():
    G = nx.DiGraph()
    
    G.add_edge('S0', 'S1', weight=0.1429, label="λ=1/7")
    G.add_edge('S1', 'S0', weight=0.1000, label="λ=1/10")
    G.add_edge('S1', 'S1', weight=1.0, label="λ=1.0 (Production)")

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#87CEEB', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, 
                           connectionstyle="arc3,rad=0.1", node_size=3000)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, label_pos=0.3)
    
    plt.title("Extended Reachability Graph (ERG)\nQuality Tester", fontsize=15)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    render_erg_diagram()