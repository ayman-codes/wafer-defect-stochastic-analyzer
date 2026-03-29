import networkx as nx
import matplotlib.pyplot as plt

def render_rate_diagram(states, generator_matrix):
    G = nx.DiGraph()
    
    for i, origin in enumerate(states):
        for j, target in enumerate(states):
            if i != j: 
                rate = generator_matrix[i][j]
                if rate > 0:
                    G.add_edge(origin, target, weight=rate, label=f"λ={rate:.4f}")

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=3, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#FFB6C1', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, 
                           connectionstyle="arc3,rad=0.1", node_size=3000)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, label_pos=0.3)
    
    plt.title("CTMC State Diagram (Rates)\nAvg Durations: S0=2min, S1=3min", fontsize=15)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    states = ["Source 0", "Source 1"]
    Q = [[-0.5, 0.5], [0.3333, -0.3333]]
    render_rate_diagram(states, Q)