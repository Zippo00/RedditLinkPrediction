import networkx as nx
from karateclub.dataset import GraphSetReader
import matplotlib.pyplot as plt
import pandas as pd

reader = GraphSetReader("reddit10k")
graphs = reader.get_graphs()
y = reader.get_target()

def task1():
    # Find the size of each graph
    graph_sizes = [len(g.nodes) for g in graphs]

    # Find the index of the smallest and largest graph
    smallest_graph_index = graph_sizes.index(min(graph_sizes))
    largest_graph_index = graph_sizes.index(max(graph_sizes))

    # Get the smallest and largest graph
    smallest_graph = graphs[smallest_graph_index]
    largest_graph = graphs[largest_graph_index]

    # Convert the smallest and largest graph to networkx graph
    smallest_network = nx.Graph(smallest_graph)
    largest_network = nx.Graph(largest_graph)

    # Print the number of nodes and edges of each graph
    print(f'Smallest network: nodes={smallest_network.number_of_nodes()}, edges={smallest_network.number_of_edges()}')
    print(f'Largest network: nodes={largest_network.number_of_nodes()}, edges={largest_network.number_of_edges()}')

    # Apply Label Propagation Algorithm to smallest network
    smallest_communities = nx.algorithms.community.label_propagation.label_propagation_communities(smallest_network)

    # Print number of communities in smallest network
    print(f'Smallest network has {len(smallest_communities)} communities')

    # Apply Label Propagation Algorithm to largest network
    largest_communities = nx.algorithms.community.label_propagation.label_propagation_communities(largest_network)

    # Print number of communities in largest network
    print(f'Largest network has {len(largest_communities)} communities')

    # Create a dictionary of communities in the largest network
    largest_communities_dict = {}
    for i, community in enumerate(largest_communities):
        for node in community:
            largest_communities_dict.setdefault(i, []).append(node)

    # Draw the largest network with communities
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(largest_network)
    for i, community_nodes in largest_communities_dict.items():
        nx.draw_networkx_nodes(largest_network, pos, nodelist=community_nodes, node_color=f'C{i}', node_size=100)
    nx.draw_networkx_edges(largest_network, pos, width=0.5)
    nx.draw_networkx_labels(largest_network, pos, font_size=6, font_family='sans-serif')
    plt.title("Largest Network with Communities")
    plt.axis('off')
    plt.show()

    # Create a dictionary of communities in the smallest network
    smallest_communities_dict = {}
    for i, community in enumerate(smallest_communities):
        for node in community:
            smallest_communities_dict.setdefault(i, []).append(node)

    # Draw the smallest network with communities
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(smallest_network)
    for i, community_nodes in smallest_communities_dict.items():
        nx.draw_networkx_nodes(smallest_network, pos, nodelist=community_nodes, node_color=f'C{i}', node_size=100)
    nx.draw_networkx_edges(smallest_network, pos, width=0.5)
    nx.draw_networkx_labels(smallest_network, pos, font_size=6, font_family='sans-serif')
    plt.title("Smallest Network with Communities")
    plt.axis('off')
    plt.show()

    # Calculate the degree distribution of the smallest network
    smallest_degree_hist = nx.degree_histogram(smallest_network)

    # Plot the degree distribution of the smallest network
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(smallest_degree_hist)), smallest_degree_hist)
    plt.title("Smallest Network Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

    # Calculate the degree distribution of the largest network
    largest_degree_hist = nx.degree_histogram(largest_network)

    # Plot the degree distribution of the largest network
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(largest_degree_hist)), largest_degree_hist)
    plt.title("Largest Network Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

def task2():
    data = {'Thread Number' : [],
            'Size': [], 
            'Diameter' : [],
            'Average Clustering Coefficient': [],
            'Highest Clustering Coefficient': [],
            'Lowest Clustering Coefficient': [],
            'Average Degree Centrality': [],
            'Highest Degree Centrality': [],
            'Lowest Degree Centrality': [],
            'Average Closeness Centrality': [],
            'Highest Closeness Centrality': [],
            'Lowest Closeness Centrality': []}
    index = 0
    for g in graphs:
            Threadnumber = index
            index += 1
            Gsize = g.size()
            Gdiameter = nx.distance_measures.diameter(g)
            Gaveragecc = nx.algorithms.cluster.average_clustering(g)
            # Dictionary of clustering coefficients of all nodes
            Gcc = nx.algorithms.cluster.clustering(g)
            # Get minimum and maximum values from dictionary
            Gminccvalue = Gcc[min(Gcc, key=Gcc.get)]
            Gmaxccvalue = Gcc[max(Gcc, key=Gcc.get)]
            # Dictionary of degree centrality of all nodes
            Gdegreecentrality = nx.algorithms.centrality.degree_centrality(g)
            Gmindcvalue = Gdegreecentrality[min(Gdegreecentrality, key=Gdegreecentrality.get)]
            Gmaxdcvalue = Gdegreecentrality[max(Gdegreecentrality, key=Gdegreecentrality.get)]
            Gavgdcvalue = sum(Gdegreecentrality.values()) / len(Gdegreecentrality)
            Gclosenesscentrality = nx.algorithms.centrality.closeness_centrality(g)
            Gminclosecentvalue = Gclosenesscentrality[min(Gclosenesscentrality, key=Gclosenesscentrality.get)]
            Gmaxclosecentvalue = Gclosenesscentrality[max(Gclosenesscentrality, key=Gclosenesscentrality.get)]
            Gavgclosecentvalue = sum(Gclosenesscentrality.values()) / len(Gclosenesscentrality)

            # Add calculated values to data dictionary
            data['Thread Number'].append(Threadnumber)
            data['Size'].append(Gsize)
            data['Diameter'].append(Gdiameter)
            data['Average Clustering Coefficient'].append(Gaveragecc)
            data['Highest Clustering Coefficient'].append(Gmaxccvalue)
            data['Lowest Clustering Coefficient'].append(Gminccvalue)
            data['Average Degree Centrality'].append(Gavgdcvalue)
            data['Highest Degree Centrality'].append(Gmaxdcvalue)
            data['Lowest Degree Centrality'].append(Gmindcvalue)
            data['Average Closeness Centrality'].append(Gavgclosecentvalue)
            data['Highest Closeness Centrality'].append(Gmaxclosecentvalue)
            data['Lowest Closeness Centrality'].append(Gminclosecentvalue)

    df = pd.DataFrame(data)
    df.to_excel('data.xlsx', index=False)




def main():
    # add functions you want to run here
    #nx.draw_networkx(g)
    #plt.show()
    task2()
if __name__ == "__main__":
    main()