import networkx as nx
import pandas as pd
import streamlit as st

@st.cache_data
def build_networkx_graph(nodes_dict_list, edges_dict_list):
    """
    Constructs a NetworkX MultiDiGraph from serializable node and edge dictionaries.
    """
    G = nx.MultiDiGraph()

    for node in nodes_dict_list:
        nid = str(node["id"])
        G.add_node(
            nid,
            name=node.get("name", nid),
            type=node.get("type", "Entity"),
            risk_score=float(node.get("risk_score", 50)),
            role=node.get("role", "Unknown"),
            location=node.get("location", "N/A"),
            details=node.get("details", "")
        )

    for edge in edges_dict_list:
        src = str(edge["source"])
        tgt = str(edge["target"])
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(
                src,
                tgt,
                type=edge.get("type", "LINKED_TO"),
                weight=float(edge.get("weight", 1.0)),
                timestamp=edge.get("timestamp", "N/A"),
                details=edge.get("details", "")
            )

    return G

def compute_graph_metrics(G):
    """
    Computes centrality algorithms (PageRank, Betweenness, Degree) and Community Detection.
    Returns enriched metrics dataframe.
    """
    if len(G) == 0:
        return pd.DataFrame()

    # Convert to simple graph for centrality algorithms
    G_simple = nx.Graph(G)

    # Centrality measures
    degree_cent = nx.degree_centrality(G_simple)
    betweenness_cent = nx.betweenness_centrality(G_simple, weight='weight')
    
    try:
        pagerank_vals = nx.pagerank(G_simple, weight='weight')
    except Exception:
        pagerank_vals = {n: 1.0 / max(len(G), 1) for n in G.nodes()}

    try:
        eigenvector_vals = nx.eigenvector_centrality(G_simple, max_iter=500)
    except Exception:
        eigenvector_vals = {n: 0.0 for n in G.nodes()}

    # Community detection
    try:
        communities = list(nx.community.greedy_modularity_communities(G_simple))
        community_map = {}
        for comm_idx, comm_nodes in enumerate(communities):
            for n in comm_nodes:
                community_map[n] = f"Cell #{comm_idx+1}"
    except Exception:
        community_map = {n: "Cell #1" for n in G.nodes()}

    metrics_list = []
    for n in G.nodes():
        node_attr = G.nodes[n]
        metrics_list.append({
            "id": n,
            "name": node_attr.get("name", n),
            "type": node_attr.get("type", "Entity"),
            "risk_score": node_attr.get("risk_score", 50),
            "degree": G_simple.degree(n),
            "degree_centrality": round(degree_cent.get(n, 0.0), 4),
            "betweenness_centrality": round(betweenness_cent.get(n, 0.0), 4),
            "pagerank": round(pagerank_vals.get(n, 0.0), 4),
            "eigenvector": round(eigenvector_vals.get(n, 0.0), 4),
            "community": community_map.get(n, "Cell #1"),
            "location": node_attr.get("location", "N/A"),
            "role": node_attr.get("role", "N/A")
        })

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.sort_values(by="betweenness_centrality", ascending=False)
    return df_metrics

def find_shortest_path(G, source_node, target_node):
    """
    Finds the shortest network path between two criminal entities.
    Returns list of node IDs in path and list of edges connecting them.
    """
    if not G.has_node(source_node) or not G.has_node(target_node):
        return None, "One or both selected nodes do not exist in the active graph."
    
    try:
        G_simple = nx.Graph(G)
        path = nx.shortest_path(G_simple, source=source_node, target=target_node)
        
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G_simple.get_edge_data(u, v)
            etype = "LINKED_TO"
            if isinstance(edge_data, dict):
                first_key = list(edge_data.keys())[0] if edge_data else None
                if first_key is not None and isinstance(edge_data[first_key], dict):
                    etype = edge_data[first_key].get("type", "LINKED_TO")
                elif "type" in edge_data:
                    etype = edge_data["type"]

            path_edges.append({
                "from": u,
                "to": v,
                "type": etype
            })
            
        return path, path_edges
    except nx.NetworkXNoPath:
        return None, f"No direct or indirect connection found between {source_node} and {target_node}."
    except Exception as e:
        return None, str(e)

def filter_subgraph(G, allowed_types=None, min_risk=0, max_nodes=500):
    """
    Filters graph based on entity types, risk score, and node limits for optimal rendering performance.
    """
    nodes_to_keep = []
    for n, data in G.nodes(data=True):
        risk = data.get("risk_score", 0)
        etype = data.get("type", "Entity")
        
        if risk >= min_risk:
            if allowed_types is None or etype in allowed_types:
                nodes_to_keep.append(n)
                
    if len(nodes_to_keep) > max_nodes:
        deg_dict = dict(G.degree(nodes_to_keep))
        sorted_nodes = sorted(nodes_to_keep, key=lambda x: (G.nodes[x].get("risk_score", 0), deg_dict[x]), reverse=True)
        nodes_to_keep = sorted_nodes[:max_nodes]

    subG = G.subgraph(nodes_to_keep).copy()
    return subG

def get_node_dossier(G, node_id):
    """
    Retrieves full suspect dossier and neighbor connection table for node inspector UI.
    """
    if not G.has_node(node_id):
        return None
        
    node_attr = G.nodes[node_id]
    G_simple = nx.Graph(G)
    
    neighbors = list(G_simple.neighbors(node_id))
    neighbor_details = []
    
    for nbr in neighbors:
        nbr_attr = G.nodes[nbr]
        edge_data = G_simple.get_edge_data(node_id, nbr)
        etype = "LINKED_TO"
        if isinstance(edge_data, dict):
            first_key = list(edge_data.keys())[0] if edge_data else None
            if first_key is not None and isinstance(edge_data[first_key], dict):
                etype = edge_data[first_key].get("type", "LINKED_TO")
            elif "type" in edge_data:
                etype = edge_data["type"]
                
        neighbor_details.append({
            "target_id": nbr,
            "target_name": nbr_attr.get("name", nbr),
            "target_type": nbr_attr.get("type", "Entity"),
            "target_risk": nbr_attr.get("risk_score", 50),
            "relation_type": etype
        })
        
    return {
        "id": node_id,
        "name": node_attr.get("name", node_id),
        "type": node_attr.get("type", "Entity"),
        "risk_score": node_attr.get("risk_score", 50),
        "role": node_attr.get("role", "N/A"),
        "location": node_attr.get("location", "N/A"),
        "details": node_attr.get("details", ""),
        "degree": G_simple.degree(node_id),
        "neighbors": neighbor_details
    }
