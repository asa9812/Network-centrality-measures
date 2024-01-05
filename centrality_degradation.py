import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
import random
from numpy.linalg import eigvals 
from scipy.linalg import inv, eigvals, block_diag 
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn is used for heatmap visualization
from scipy.stats import skew
import scipy.stats as stats
from scipy.special import comb
import sympy as sp
import matplotlib.pyplot as plt

df = pd.read_csv('downloadSupplement.txt', sep='\t', header=None)

df.columns = ['city', 'node1', 'node2'] 

city_centrality_matrices_fp_NBTW = {}  
city_centrality_matrices_fp_katz = {} 
city_centrality_matrices_fn_NBTW = {}
city_centrality_matrices_fn_katz = {} 

city_spectral_diffs_katz_fn ={}
city_spectral_diffs_katz_fp ={}

city_spectral_diffs_nbtw_fn ={}
city_spectral_diffs_nbtw_fp ={}

city_frobenius_diffs_katz_fn ={}
city_frobenius_diffs_katz_fp ={}

city_frobenius_diffs_nbtw_fn ={}
city_frobenius_diffs_nbtw_fp ={}

city_avg_abs_errors_katz_centrality_fp = {}
city_mean_squared_errors_katz_centrality_fp = {}

city_avg_abs_errors_katz_centrality_fn = {}
city_mean_squared_errors_katz_centrality_fn = {}

city_avg_abs_errors_nbtw_centrality_fp = {}
city_mean_squared_errors_nbtw_centrality_fp = {}

city_avg_abs_errors_nbtw_centrality_fn = {}
city_mean_squared_errors_nbtw_centrality_fn = {}


def calculate_spectral_radius(G):
    A = nx.adjacency_matrix(G).toarray()  
    return max(abs(eigvals(A)))

def nbtw_centrality_matrix(G, alpha):
    A = nx.adjacency_matrix(G).toarray()
    D = np.diag(np.sum(A, axis=0))  
    I = np.eye(len(A))  

    # Compute the matrix polynomial M(alpha)
    M_alpha = I - alpha * A + (alpha**2) * (D - I)

    # Calculate the NBTW centrality matrix Q
    Q = (1 - alpha**2) * inv(M_alpha)

    return Q 

def number_of_combinations(G, i):
    return comb(num_nodes(G), i)

def katz_centrality_matrix(G, alpha):
    A = nx.adjacency_matrix(G).toarray()  
    I = np.eye(len(A)) 
    Q = inv(I - alpha*A)
    return Q

def num_edges(G):
    return G.number_of_edges()

def num_nodes(G):
    return G.number_of_nodes()  


def add_false_positives(G, percentage):
    
    non_edges = list(nx.non_edges(G))
    num_false_positives = int(len(non_edges) * percentage)
    false_positives = random.sample(non_edges, num_false_positives)
    G_with_false_positives = G.copy()
    G_with_false_positives.add_edges_from(false_positives)
    return G_with_false_positives

def remove_false_negatives(G, percentage):
    true_edges = list(G.edges()) 
    num_false_negatives = int(len(G.edges()) * percentage) 
    false_negatives = random.sample(true_edges, num_false_negatives)
    G_with_false_negatives = G.copy()
    G_with_false_negatives.remove_edges_from(false_negatives) 
    return G_with_false_negatives


def calculate_aae(true_matrix, error_matrix):
    return np.mean(np.abs(true_matrix - error_matrix))


def calculate_mse(true_matrix, error_matrix):
    return np.mean((true_matrix - error_matrix)**2)


def frobenius_norm_difference(A, B):
    return np.linalg.norm(A - B, 'fro')

def spectral_norm(A):
    return np.linalg.norm(A, 2)

def spectral_norm_difference(A, B):
    return np.linalg.norm(A - B, 2)


def extract_data_for_plotting(data_dict, city):
    percentages = []
    values = []
    for (city_key, percentage), value in data_dict.items():
        if city_key == city:
            percentages.append(float(percentage))
            values.append(value)
    return percentages, values

def create_plot(city, norm_type, title):
    fig, ax = plt.subplots()

    if norm_type == 'Frobenius':
        fn_dict = city_frobenius_diffs_katz_fn
        fp_dict = city_frobenius_diffs_katz_fp
        fn_dict_nbtw = city_frobenius_diffs_nbtw_fn
        fp_dict_nbtw = city_frobenius_diffs_nbtw_fp
    elif norm_type == 'Spectral':
        fn_dict = city_spectral_diffs_katz_fn
        fp_dict = city_spectral_diffs_katz_fp
        fn_dict_nbtw = city_spectral_diffs_nbtw_fn
        fp_dict_nbtw = city_spectral_diffs_nbtw_fp

    # Plot the data for Katz matrices
    x, y = extract_data_for_plotting(fn_dict, city)
    ax.plot(x, y, label=f'{norm_type} Katz FN')

    x, y = extract_data_for_plotting(fp_dict, city)
    ax.plot(x, y, label=f'{norm_type} Katz FP')

    # Plot the data for NBTW matrices
    x, y = extract_data_for_plotting(fn_dict_nbtw, city)
    ax.plot(x, y, label=f'{norm_type} NBTW FN')

    x, y = extract_data_for_plotting(fp_dict_nbtw, city)
    ax.plot(x, y, label=f'{norm_type} NBTW FP')

    # Labeling and titling the plot
    ax.set_xlabel('fn/fp')
    ax.set_ylabel('Difference')
    ax.set_title(title)
    ax.legend()

    return fig, ax


def create_plot_2(city, statistic_type, title):
    fig, ax = plt.subplots()

    # Depending on the norm_type, select the appropriate dictionaries
    if statistic_type == 'MSE':
        fn_dict = city_mean_squared_errors_katz_centrality_fn
        fp_dict = city_mean_squared_errors_katz_centrality_fp
        fn_dict_nbtw = city_mean_squared_errors_nbtw_centrality_fn
        fp_dict_nbtw = city_mean_squared_errors_nbtw_centrality_fp
    elif statistic_type == 'Average absolute error':
        fn_dict = city_avg_abs_errors_katz_centrality_fn
        fp_dict = city_avg_abs_errors_katz_centrality_fp
        fn_dict_nbtw = city_avg_abs_errors_nbtw_centrality_fn
        fp_dict_nbtw = city_avg_abs_errors_nbtw_centrality_fp
    # Plot the data for Katz matrices
    x, y = extract_data_for_plotting(fn_dict, city)
    ax.plot(x, y, label=f'{statistic_type} Katz FN')

    x, y = extract_data_for_plotting(fp_dict, city)
    ax.plot(x, y, label=f'{statistic_type} Katz FP')

    # Plot the data for NBTW matrices
    x, y = extract_data_for_plotting(fn_dict_nbtw, city)
    ax.plot(x, y, label=f'{statistic_type} NBTW FN')

    x, y = extract_data_for_plotting(fp_dict_nbtw, city)
    ax.plot(x, y, label=f'{statistic_type} NBTW FP')

    # Labeling and titling the plot
    ax.set_xlabel('fn/fp')
    ax.set_ylabel('Error value')
    ax.set_title(title)
    ax.legend()

    return fig, ax


# Process each city's data
for city in df['city'].unique():
    # Filter the DataFrame for the current city
    city_df = df[df['city'] == city]
    
    # Create a graph for the current city
    G = nx.Graph()
    G.add_edges_from(city_df[['node1', 'node2']].values)

    # Calculate the spectral radius for the original graph
    rho = calculate_spectral_radius(G) 
    
 #_________________________________________________________________________________#
    A = nx.adjacency_matrix(G).toarray()

    # Convert A to a floting-point data type 
    A = A.astype(float)
        
    D = np.diag(A.sum(axis=1))

    I = np.eye(A.shape[0])

    shape_of_A = A.shape
    zero_matrix = np.zeros(shape_of_A)

    C = np.bmat([[A, I - D],
                    [I, zero_matrix]])


   
    #NBTW spectral radius calculations:
    eigenvalues_C = np.linalg.eigvals(C)

    # Find the largest eigenvalue in modulus (spectral radius)
    spectral_radius_C = max(abs(eigenvalues_C))

    # Calculate the reciprocal of the largest eigenvalue in modulus
    rho_nbtw = spectral_radius_C
  #__________________________________________________________________#

    # percentage of true edges to remove (or percentage of absent edges to introduce)
    percentages = np.linspace(0.000001, 0.9, num=6)  

    
    nbtw_centrality_fp_matrices=[] 
    katz_centrality_fp_matrices=[]
    nbtw_centrality_fn_matrices=[]
    katz_centrality_fn_matrices=[]
    
    # Initialize arrays to store the results
    frobenius_diffs_katz_fp = []
    spectral_diffs_katz_fp = []

    frobenius_diffs_katz_fn = []
    spectral_diffs_katz_fn = []

    frobenius_diffs_nbtw_fp = []
    spectral_diffs_nbtw_fp = []

    frobenius_diffs_nbtw_fn = []
    spectral_diffs_nbtw_fn = []
    
    
    avg_abs_errors_katz_centrality_fp = []
    mean_squared_errors_katz_centrality_fp = []
    
    avg_abs_errors_katz_centrality_fn = []
    mean_squared_errors_katz_centrality_fn = []
    
    avg_abs_errors_nbtw_centrality_fp = []
    mean_squared_errors_nbtw_centrality_fp = []
    
    avg_abs_errors_nbtw_centrality_fn = []
    mean_squared_errors_nbtw_centrality_fn = []
    
    n = num_nodes(G) 
    n_2 = num_edges(G)
    num_potential_edges = n * (n - 1) / 2
    
    
    true_katz_centrality_matrix = katz_centrality_matrix(G, alpha=0.8/rho)
        
    true_nbtw_centrality_matrix = nbtw_centrality_matrix(G, alpha=0.8/rho_nbtw)
        
  
    for percentage in percentages:
     
        G_fp = add_false_positives(G.copy(), percentage)
        G_fn = remove_false_negatives(G.copy(), percentage)
        
        rho_fp = calculate_spectral_radius(G_fp)
        
        rho_fn = calculate_spectral_radius(G_fn)
        
        
        #Introduct A of both type fp and fn.
        A_fn = nx.adjacency_matrix(G_fn).toarray()

        # Convert A to a floting-point data type 
        A_fn = A_fn.astype(float)
        
        
        D_fn = np.diag(A_fn.sum(axis=1))
         
        # Identity matrix I of the same dimension as A
        
        I = np.eye(A_fn.shape[0])
        
        shape_of_A_fn = A_fn.shape
        zero_matrix_fn = np.zeros(shape_of_A_fn)
        
        
        C_fn = np.bmat([[A_fn, I - D_fn],
                        [I, zero_matrix_fn]])
        
      
        #NBTW spectral radius calucatations:
        eigenvalues_C_fn = np.linalg.eigvals(C_fn)
        # Find the largest eigenvalue in modulus (spectral radius)
        spectral_radius_C_fn = max(abs(eigenvalues_C_fn))

        # Calculate the reciprocal of the largest eigenvalue in modulus
        rho_nbtw_fn = spectral_radius_C_fn
        print(f"for city {city}, spectral radius of C for fn at {percentage} is {rho_nbtw_fn}") 
  #____________________________________________________________________________________________________#
    
        #Introduct A of both type fp and fn.
        A_fp = nx.adjacency_matrix(G_fp).toarray()
      
        # Convert A to a floting-point data type 
        A_fp = A_fp.astype(float)
               
 
        D_fp = np.diag(A_fp.sum(axis=1)) # THIS SUMS THE ELEMTS ON EACH ROW TO CREATE THE DIAG. NODAL DEGREE MATRIX.
        
        # Identity matrix I of the same dimension as A
        I = np.eye(A_fp.shape[0])
       
        shape_of_A_fp = A_fp.shape
        zero_matrix_fp = np.zeros(shape_of_A_fp)
       
        C_fp = np.bmat([[A_fp, I - D_fp],
                        [I, zero_matrix_fp]])
        
 #____________________________________________________________________________________________________#
    
        #NBTW spectral radius calucatations:
        eigenvalues_C_fp = np.linalg.eigvals(C_fp)
        # Find the largest eigenvalue in modulus (spectral radius)
        spectral_radius_C_fp = max(abs(eigenvalues_C_fp))

        # Calculate the reciprocal of the largest eigenvalue in modulus
        rho_nbtw_fp = spectral_radius_C_fp
        print(f"for city {city}, spectral radius of C for fp at {percentage} is {rho_nbtw_fp}") 
        
 #____________________________________________________________________________________________________#
        
        
        # Calculate Katz centrality for each modified graph
        katz_centrality_fp = katz_centrality_matrix(G_fp, alpha=0.8/rho_fp)
        katz_centrality_fp_matrices.append(katz_centrality_fp)
        
        katz_centrality_fn = katz_centrality_matrix(G_fn, alpha=0.8/rho_fn) # pot. abck to numpy
        katz_centrality_fn_matrices.append(katz_centrality_fn)
        
        
        # Calculate NBTW centrality for each modified graph
        nbtw_centrality_fp = nbtw_centrality_matrix(G_fp, alpha=0.8/rho_nbtw_fp) 
        nbtw_centrality_fp_matrices.append(nbtw_centrality_fp)
        
        nbtw_centrality_fn = nbtw_centrality_matrix(G_fn, alpha=0.8/rho_nbtw_fn)
        nbtw_centrality_fn_matrices.append(nbtw_centrality_fn)
        
 #_____________________________________________________________________________________________________#
    
        # Calculate the errors using the sample data
        average_absolute_error_katz_centrality_fp = calculate_aae(true_katz_centrality_matrix, katz_centrality_fp)
        mean_squared_error_katz_centrality_fp = calculate_mse(true_katz_centrality_matrix, katz_centrality_fp)

        # Store the calculated errors
        avg_abs_errors_katz_centrality_fp.append(average_absolute_error_katz_centrality_fp)
        mean_squared_errors_katz_centrality_fp.append(mean_squared_error_katz_centrality_fp)
        
 
  #_____________________________________________________________________________________________________#
            
        # Calculate the errors using the sample data
        average_absolute_error_katz_centrality_fn = calculate_aae(true_katz_centrality_matrix, katz_centrality_fn)
        mean_squared_error_katz_centrality_fn = calculate_mse(true_katz_centrality_matrix, katz_centrality_fn)
        
        # Store the calculated errors
        avg_abs_errors_katz_centrality_fn.append(average_absolute_error_katz_centrality_fn)
        mean_squared_errors_katz_centrality_fn.append(mean_squared_error_katz_centrality_fn)
        
 #_____________________________________________________________________________________________________#
        
        # Calculate the errors using the sample data
        average_absolute_error_nbtw_centrality_fp = calculate_aae(true_nbtw_centrality_matrix, nbtw_centrality_fp)
        mean_squared_error_nbtw_centrality_fp = calculate_mse(true_nbtw_centrality_matrix, nbtw_centrality_fp)
        
        # Store the calculated errors
        avg_abs_errors_nbtw_centrality_fp.append(average_absolute_error_nbtw_centrality_fp)
        mean_squared_errors_nbtw_centrality_fp.append(mean_squared_error_nbtw_centrality_fp)
        
    
        # Calculate the errors using the sample data
        average_absolute_error_nbtw_centrality_fn = calculate_aae(true_nbtw_centrality_matrix, nbtw_centrality_fn)
        mean_squared_error_nbtw_centrality_fn = calculate_mse(true_nbtw_centrality_matrix, nbtw_centrality_fn)
        
        # Store the calculated errors
        avg_abs_errors_nbtw_centrality_fn.append(average_absolute_error_nbtw_centrality_fn)
        mean_squared_errors_nbtw_centrality_fn.append(mean_squared_error_nbtw_centrality_fn)
        

        # Calculate the norm differences
        frob_diff_katz_fp = frobenius_norm_difference(true_katz_centrality_matrix, katz_centrality_fp)
        spec_diff_katz_fp = spectral_norm_difference(true_katz_centrality_matrix, katz_centrality_fp)
       
        frob_diff_katz_fn = frobenius_norm_difference(true_katz_centrality_matrix, katz_centrality_fn)
        spec_diff_katz_fn = spectral_norm_difference(true_katz_centrality_matrix, katz_centrality_fn)
        
        frob_diff_nbtw_fp = frobenius_norm_difference(true_nbtw_centrality_matrix, nbtw_centrality_fp)
        spec_diff_nbtw_fp = spectral_norm_difference(true_nbtw_centrality_matrix, nbtw_centrality_fp)
        
        frob_diff_nbtw_fn = frobenius_norm_difference(true_nbtw_centrality_matrix, nbtw_centrality_fn)
        spec_diff_nbtw_fn = spectral_norm_difference(true_nbtw_centrality_matrix, nbtw_centrality_fn)
        
        #Normalize the differences by the number of potential edges (for an undirected graph without self-loops)
        norm_frob_diff_katz_fp = frob_diff_katz_fp / (n_2/num_potential_edges)
        norm_spec_diff_katz_fp = spec_diff_katz_fp / (n_2/num_potential_edges)
        
        norm_frob_diff_katz_fn = frob_diff_katz_fn / (n_2/num_potential_edges)
        norm_spec_diff_katz_fn = spec_diff_katz_fn / (n_2/num_potential_edges)
        
        norm_frob_diff_nbtw_fp = frob_diff_nbtw_fp / (n_2/num_potential_edges)
        norm_spec_diff_nbtw_fp = spec_diff_nbtw_fp / (n_2/num_potential_edges)
        
        norm_frob_diff_nbtw_fn = frob_diff_nbtw_fn / (n_2/num_potential_edges)
        norm_spec_diff_nbtw_fn = spec_diff_nbtw_fn / (n_2/num_potential_edges)

        # Append to results
        frobenius_diffs_katz_fp.append(norm_frob_diff_katz_fp)
        spectral_diffs_katz_fp.append(norm_spec_diff_katz_fp)
        
        frobenius_diffs_katz_fn.append(norm_frob_diff_katz_fn)
        spectral_diffs_katz_fn.append(norm_spec_diff_katz_fn)
        
        frobenius_diffs_nbtw_fp.append(norm_frob_diff_nbtw_fp)
        spectral_diffs_nbtw_fp.append(norm_spec_diff_nbtw_fp)
        
        frobenius_diffs_nbtw_fn.append(norm_frob_diff_nbtw_fn)
        spectral_diffs_nbtw_fn.append(norm_spec_diff_nbtw_fn)
        
    for i, percentage in enumerate(percentages):
        city_spectral_diffs_katz_fp[(city, percentage)]  = spectral_diffs_katz_fp[i]
        city_spectral_diffs_katz_fn[(city, percentage)]  = spectral_diffs_katz_fn[i]
        city_spectral_diffs_nbtw_fp[(city, percentage)]  = spectral_diffs_nbtw_fp[i]
        city_spectral_diffs_nbtw_fn[(city, percentage)]  = spectral_diffs_nbtw_fn[i]
        city_frobenius_diffs_nbtw_fn[(city, percentage)]  = frobenius_diffs_nbtw_fn[i]
        city_frobenius_diffs_nbtw_fp[(city, percentage)]  = frobenius_diffs_nbtw_fp[i]
        city_frobenius_diffs_katz_fn[(city, percentage)]  = frobenius_diffs_katz_fn[i]
        city_frobenius_diffs_katz_fp[(city, percentage)]  = frobenius_diffs_katz_fp[i]
        
     
    for i, percentage in enumerate(percentages):
        city_avg_abs_errors_katz_centrality_fp[(city, percentage)] = avg_abs_errors_katz_centrality_fp[i]
        city_mean_squared_errors_katz_centrality_fp[(city, percentage)] = mean_squared_errors_katz_centrality_fp[i]
        city_avg_abs_errors_katz_centrality_fn[(city, percentage)] = avg_abs_errors_katz_centrality_fn[i]
        city_mean_squared_errors_katz_centrality_fn[(city, percentage)] = mean_squared_errors_katz_centrality_fn[i]
        city_avg_abs_errors_nbtw_centrality_fp[(city, percentage)] = avg_abs_errors_nbtw_centrality_fp[i]
        city_mean_squared_errors_nbtw_centrality_fp[(city, percentage)] = mean_squared_errors_nbtw_centrality_fp[i]
        city_avg_abs_errors_nbtw_centrality_fn[(city, percentage)] = avg_abs_errors_nbtw_centrality_fn[i]
        city_mean_squared_errors_nbtw_centrality_fn[(city, percentage)] = mean_squared_errors_nbtw_centrality_fn[i]
   
 #_____________________________________________________________________________________________________#
    norm_types = ['Frobenius', 'Spectral']


    for norm_type in norm_types:
        create_plot(city, norm_type, f'Normalized {norm_type} Differences for city {city}')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
    
    statistic_types = ['MSE', 'Average absolute error']
    
    for statistic_type in statistic_types:
        create_plot_2(city, statistic_type, f'{statistic_type} of differences for city {city}')

    plt.tight_layout()
    plt.show()
