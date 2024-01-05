import pandas as pd 
import networkx as nx # renaming libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn is used for heatmap visualization
from scipy.linalg import inv, eigvals 
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.stats import skew, kurtosis, pearsonr, kendalltau

# Load the network data
df = pd.read_csv('downloadSupplement.txt', delimiter='\t', header=None)

# Assuming the first column is the city index, and the next two columns are the nodes
df.columns = ['city', 'node1', 'node2']

# Function to calculate the spectral radius of a graph
def calculate_spectral_radius(G):
    A = nx.adjacency_matrix(G).astype('d')
    A = csr_matrix(A)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_edges = n * (n - 1) / 2
    sparsity_ratio = m / max_edges

    if sparsity_ratio < 0.1:  # Adjust threshold as req.
        # Sparse method
        eigenvalues, _ = eigs(A, k=1, which='LM')
        spectral_radius = abs(eigenvalues[0])
    else:
        # Dense method
        A_dense = A.toarray() 
        eigenvalues = eigvals(A_dense)
        spectral_radius = max(abs(eigenvalues))

    return spectral_radius

# Function to calculate the NBTW centrality matrix
def nbtw_centrality_matrix(G, alpha):
    # Create the adjacency matrix A and the degree matrix D
    A = nx.adjacency_matrix(G).toarray()  
    D = np.diag(np.sum(A, axis=0))  # Diagonal degree matrix
    I = np.eye(len(A))  # Identity matrix

    # Compute the matrix polynomial M(alpha)
    M_alpha = I - alpha * A + alpha**2 * (D - I)

    # Calculate the NBTW centrality matrix Q
    Q = (1 - alpha**2) * inv(M_alpha)

    return Q

def katz_centrality_matrix(G, alpha):
    # Create the adjacency matrix A and the degree matrix D
    A = nx.adjacency_matrix(G).toarray()  
    I = np.eye(len(A))  # Identity matrix
    # Calculate the NBTW centrality matrix Q
    Q = inv(I - alpha*A)
    return Q


def get_katz_centrality_scores(Q):
    # Since Q includes the identity matrix, subtract it to get the centrality contributions
    centrality_matrix = Q - np.eye(len(Q))
    # Sum across rows to get the Katz centrality scores
    katz_centrality_scores = np.sum(centrality_matrix, axis=1)
    return katz_centrality_scores


def get_nbtw_centrality_scores(Q):
    nbtw_centrality_scores = np.sum(Q, axis=1)
    return nbtw_centrality_scores



nbtw_centrality_matrices = {}  
katz_centrality_matrices = {} 

# Process each city's data
for city in df['city'].unique(): 

    city_edges = df[df['city'] == city][['node1', 'node2']]

    # Create a graph for the current city
    G = nx.Graph()
    G.add_edges_from(city_edges.values)

    # Print basic properties of the graph
    print(f"City {city}:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    average_degree = np.mean([degree for node, degree in G.degree()])
    print(f"Average degree: {average_degree}")

    # Visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=False, node_size=50, font_size=8)
    plt.title(f"Network Graph for City {city}")
    plt.show()

    # Calculate the spectral radius of the adjacency matrix for each city
    A = nx.adjacency_matrix(G).toarray()
    A=A.astype(float) 
    rho = calculate_spectral_radius(G)
    print(f"spectral radius of A: {rho} for city {city}") 

    alpha_supremum = 1 / rho


    num_alphas = 3

    alphas = np.linspace((alpha_supremum*0.9)/1000, alpha_supremum * 0.9, num_alphas)



    D = np.diag(A.sum(axis=1))


    I = np.eye(A.shape[0])

    C = np.block([ 
    [A, I - D],
    [I, np.zeros_like(A)]])

    #NBTW spectral radius calucatations:

    eigenvalues_C = np.linalg.eigvals(C)

    spectral_radius_C = max(abs(eigenvalues_C))


    rho_nbtw = spectral_radius_C
    print(f"spectral radius of C: {rho_nbtw} for city {city}") 

    #alpha_nbtw_infimum = 0 

    alpha_nbtw_supremum = 1 / rho_nbtw

    num_alphas = 3
    alphas_nbtw = np.linspace((alpha_nbtw_supremum*0.9)/1000, alpha_nbtw_supremum * 0.9, num_alphas)
    print(alphas_nbtw)

    nbtw_city_matrices = []
    katz_city_matrices = []


    for alpha_value in alphas:

        katz_centrality = katz_centrality_matrix(G, alpha_value)
        katz_city_matrices.append(katz_centrality)



    for i, alpha_value in enumerate(alphas):

        katz_centrality_matrices[(city, alpha_value)] = katz_city_matrices[i]


    for alpha_value_2 in alphas_nbtw:

        Q_nbtw = nbtw_centrality_matrix(G, alpha_value_2)
        nbtw_city_matrices.append(Q_nbtw)

    for i, alpha_value_2 in enumerate(alphas_nbtw):

        nbtw_centrality_matrices[(city, alpha_value_2)] = nbtw_city_matrices[i]


    kendalls_tau_results = {}

    for i in range(num_alphas):
        katz_alpha = alphas[i]
        nbtw_alpha = alphas_nbtw[i]

    # Compute centrality matrices for this pair of alphas (Placeholder)
        Q_katz = katz_centrality_matrix(G, katz_alpha) 
        Q_nbtw = nbtw_centrality_matrix(G, nbtw_alpha)

    # Calculate centrality scores
        katz_scores = get_katz_centrality_scores(Q_katz)
        nbtw_scores = get_nbtw_centrality_scores(Q_nbtw)
        top_katz_indices = np.argsort(katz_scores)[-100:]
        top_nbtw_scores = nbtw_scores[top_katz_indices]
    # Calculate Kendall's tau
        tau, p_value = kendalltau(katz_scores[top_katz_indices], top_nbtw_scores) # Kendall's tau for top 100 Katz ranked nodes.


        kendalls_tau_results[(katz_alpha, nbtw_alpha)] = (tau, p_value)

    for (katz_alpha, nbtw_alpha), (tau, p_value) in kendalls_tau_results.items():
         print(f"City: {city}, Katz alpha: {katz_alpha}, NBTW alpha: {nbtw_alpha}, Kendall's tau: {tau}, p-value: {p_value}")
