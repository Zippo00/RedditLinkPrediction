"""
Social Network Analysis Course Project on Reddit Data Link Analysis tasks 1-4
Authors: Aleksanteri Kylmäaho & Joonas Kelloniemi 2023
"""


import networkx as nx
from karateclub.dataset import GraphSetReader
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import copy

reader = GraphSetReader("reddit10k")
graphs = reader.get_graphs()
y = reader.get_target()

def task1():
    print("--- Task 1 ---")
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

# Despite the name, also contains tasks 3 and 4.
def task2():
    print("--- Task 2 ---")

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
    # Copy the dictionary template for task 4
    discussiondata = copy.deepcopy(data)
    nondiscussiondata = copy.deepcopy(data)

    # Turns a list of graphs into a dictionary with attributes listed
    # Dictionary must be defined in advance
    def graphs_to_dictionary(graphs_list,dictionary):
        index = 0
        for g in graphs_list:
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
                dictionary['Thread Number'].append(Threadnumber)
                dictionary['Size'].append(Gsize)
                dictionary['Diameter'].append(Gdiameter)
                dictionary['Average Clustering Coefficient'].append(Gaveragecc)
                dictionary['Highest Clustering Coefficient'].append(Gmaxccvalue)
                dictionary['Lowest Clustering Coefficient'].append(Gminccvalue)
                dictionary['Average Degree Centrality'].append(Gavgdcvalue)
                dictionary['Highest Degree Centrality'].append(Gmaxdcvalue)
                dictionary['Lowest Degree Centrality'].append(Gmindcvalue)
                dictionary['Average Closeness Centrality'].append(Gavgclosecentvalue)
                dictionary['Highest Closeness Centrality'].append(Gmaxclosecentvalue)
                dictionary['Lowest Closeness Centrality'].append(Gminclosecentvalue)

    graphs_to_dictionary(graphs,data)
    # Push data to excel file
    df = pd.DataFrame(data)
    df.to_excel('data.xlsx', index=False)

    print("--- Task 3 ---")

    # Give size list and chosen attribute list as parameters.
    # Returns ordered size list for x-axis and parameter list for y-axis, with duplicate sizes removed
    def size_and_attribute(size_list, attribute_list):
        size_list_original = size_list.copy()
        attribute_list_original = attribute_list.copy()
        attribute_data = {}
        while size_list_original:
            iterable_size = size_list_original.pop(0)
            attribute_data[iterable_size] = 0
            avg_attribute = []
            avg_attribute.append(attribute_list_original.pop(0))
            while iterable_size in size_list_original:
                size_index = size_list_original.index(iterable_size)
                size_list_original.pop(size_index)
                avg_attribute.append(attribute_list_original.pop(size_index))
            attribute_avg = np.average(avg_attribute)
            attribute_std_deviation = np.std(avg_attribute)
            attribute_data[iterable_size] = attribute_avg
        
        myKeys = list(attribute_data.keys())
        myKeys.sort()
        sorted_dict = {i: attribute_data[i] for i in myKeys}
        x = sorted_dict.keys()
        y = sorted_dict.values()
        return x, y
    
    # Plot Size and attributes from chosen dictionary to graph and draw the graph.
    def plot_size_and_attributes(dictionary, title):
        size, diameter_y = size_and_attribute(dictionary['Size'], dictionary['Diameter'])
        size1, adc_y = size_and_attribute(dictionary['Size'], dictionary['Average Degree Centrality'])
        size1, acco_y = size_and_attribute(dictionary['Size'], dictionary['Average Closeness Centrality'])
        size1, acluc_y = size_and_attribute(dictionary['Size'], dictionary['Average Clustering Coefficient'])
        plt.plot(size, diameter_y, label='Diameter')
        plt.plot(size, adc_y, label='Average Degree Centrality')
        plt.plot(size, acco_y, label='Average Closeness Centrality')
        plt.plot(size, acluc_y, label='Average Clustering Coefficient')
        plt.xlabel('Size')
        plt.ylabel('Attribute')
        plt.title('Attribute variance in relation to size for ' + title)
        plt.legend()
        plt.show()
    
    plot_size_and_attributes(data, "all threads")

    # Extract pearson values from dictionary and push them to excel file. Define .xlsx filename as parameter.
    def get_pearsondata(dictionary, filename):
        corr_dia, pval_dia = pearsonr(dictionary['Size'], dictionary['Diameter'])
        corr_adc, pval_adc = pearsonr(dictionary['Size'], dictionary['Average Degree Centrality'])
        corr_accen, pval_accen = pearsonr(dictionary['Size'], dictionary['Average Closeness Centrality'])
        corr_acco, pval_acco = pearsonr(dictionary['Size'], dictionary['Average Clustering Coefficient'])
        print(corr_dia)
        print(pval_dia)
        print(corr_adc)
        print(pval_adc)
        print(corr_accen)
        print(pval_accen)
        print(corr_acco)
        print(pval_acco)
        # Pearson data to dataframe and save to excel file
        pearsondata = {
            "Diameter" : [corr_dia,pval_dia],
            'Average Degree Centrality' : [corr_adc, pval_adc], 
            'Average Closeness Centrality' : [corr_accen, pval_accen], 
            'Average Clustering Coefficient' : [corr_acco, pval_acco]
        }
        dfpearson = pd.DataFrame(pearsondata)
        dfpearson.to_excel(filename, index=False)

    print("--Pearson data for All threads--")
    get_pearsondata(data, "pearsondata_all.xlsx")

    print("---- Task 4 ----")
    # Split graphs into discussion and non-discussion
    graphs_discussion = []
    graphs_nondiscussion = []
    index = 0
    for g in graphs:
        if y[index] == 1:
            graphs_discussion.append(g)
        else:
            graphs_nondiscussion.append(g)
        index += 1
    
    # Discussiondata and nondiscussiondata were defined in task 2 as copies of the empty "data" dictionary template
    # Push Discussion and Non-Discussion graph data to respective dictionaries
    graphs_to_dictionary(graphs_discussion, discussiondata)
    graphs_to_dictionary(graphs_nondiscussion, nondiscussiondata)
    # Repeat Task 3 for these dictionaries
    plot_size_and_attributes(discussiondata, "discussion threads")
    plot_size_and_attributes(nondiscussiondata, "non-discussion threads")
    print("--Pearson data for Discussion threads--")
    get_pearsondata(discussiondata, "pearsondata_discussion.xlsx")
    print("--Pearson data for Non-Discussion threads--")
    get_pearsondata(nondiscussiondata, "pearsondata_nondiscussion.xlsx")


def main():
    task1()
    task2()
if __name__ == "__main__":
    main()