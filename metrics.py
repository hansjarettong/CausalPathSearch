import numpy as np
from scipy.stats import kendalltau
import networkx as nx

def count_shd(B_true, B_est):
    """
    Calculates the Structural Hamming Distance (SHD) between two adjacency matrices.
    SHD is the number of edge additions, deletions, or reversals needed to
    transform one graph into the other.

    Parameters
    ----------
    B_true : np.ndarray
        The ground truth adjacency matrix.
    B_est : np.ndarray
        The estimated adjacency matrix.

    Returns
    -------
    int
        The Structural Hamming Distance.
    """
    B_true_binary = (B_true != 0).astype(int)
    B_est_binary = (B_est != 0).astype(int)

    # Difference matrix
    diff = B_true_binary - B_est_binary

    # Edges to be added (present in true, not in est)
    additions = np.sum(diff == 1)
    # Edges to be deleted (present in est, not in true)
    deletions = np.sum(diff == -1)

    # Count reversals
    reversals = np.sum((B_true_binary.T == 1) & (diff == -1))

    # SHD = (additions - reversals) + (deletions - reversals) + reversals
    # SHD = additions + deletions - reversals
    shd = additions + deletions - reversals
    
    return shd


def calculate_kendall_tau(true_order, estimated_order):
    """
    Calculates Kendall's Tau correlation coefficient between two causal orders.

    Parameters
    ----------
    true_order : list or np.ndarray
        The ground truth causal order.
    estimated_order : list or np.ndarray
        The estimated causal order.

    Returns
    -------
    float
        The Kendall's Tau correlation coefficient.
    """
    tau, _ = kendalltau(true_order, estimated_order)
    return tau

def calculate_sid(B_true, B_est):
    """
    Calculates the Structural Intervention Distance (SID) between two graphs.
    The SID counts the number of pairs of nodes (i, j) for which the causal
    influence of i on j is different between the true and estimated graphs.
    A lower SID indicates a better match.

    Parameters
    ----------
    B_true : np.ndarray
        The ground truth adjacency matrix.
    B_est : np.ndarray
        The estimated adjacency matrix.

    Returns
    -------
    int
        The Structural Intervention Distance.
    """
    # Our adjacency matrix convention is that B[i, j] != 0 means an edge j -> i.
    # The networkx library convention is that M[i, j] != 0 means an edge i -> j.
    # Therefore, we must transpose our matrices before creating the graphs.
    G_true = nx.from_numpy_array(B_true.T, create_using=nx.DiGraph)
    G_est = nx.from_numpy_array(B_est.T, create_using=nx.DiGraph)

    # The transitive closure of a graph contains an edge (i, j) if and only if
    # there is a directed path from i to j in the original graph.
    TC_true_graph = nx.transitive_closure(G_true, reflexive=False)
    TC_est_graph = nx.transitive_closure(G_est, reflexive=False)

    # Convert the transitive closure graphs back to adjacency matrices.
    # The node order is preserved.
    TC_true_matrix = nx.to_numpy_array(TC_true_graph, nodelist=range(len(B_true)))
    TC_est_matrix = nx.to_numpy_array(TC_est_graph, nodelist=range(len(B_est)))

    # The SID is the number of differing entries in the transitive closure matrices.
    # This is equivalent to the size of the symmetric difference of the descendant sets.
    sid = np.sum(TC_true_matrix != TC_est_matrix)
    
    return int(sid)

