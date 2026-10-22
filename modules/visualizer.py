import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import streamlit as st

ENTITY_COLOR_MAP = {
    "Person": "#ff2a6d",        # Cyber Neon Pink/Red
    "Phone": "#00f3ff",         # Cyber Neon Cyan
    "Bank Account": "#a855f7",  # Cyber Neon Purple
    "Vehicle": "#ffaa00",       # Cyber Orange
    "Location": "#00ff66",      # Cyber Neon Green
    "FIR Case File": "#3b82f6", # Cyber Neon Blue
    "Entity": "#cbd5e1"         # Slate Silver
}

RISK_COLOR_MAP = {
    "Critical (85-100)": "#ff2a6d",
    "High (70-84)": "#ffaa00",
    "Medium (50-69)": "#00f3ff",
    "Low (0-49)": "#00ff66"
}

CAMERA_PRESETS = {
    "Isometric": dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    "Top View": dict(eye=dict(x=0.01, y=0.01, z=2.5), up=dict(x=0, y=1, z=0)),
    "Front View": dict(eye=dict(x=0.0, y=2.5, z=0.0)),
    "Macro Focus": dict(eye=dict(x=0.8, y=0.8, z=0.6))
}

def get_node_color_by_risk(score):
    if score >= 85:
        return "#ff2a6d"
    elif score >= 70:
        return "#ffaa00"
    elif score >= 50:
        return "#00f3ff"
    else:
        return "#00ff66"

@st.cache_data
def generate_3d_network_plot(nodes_dict_list, edges_dict_list, color_by="type", highlight_path=None, camera_preset="Isometric"):
    """
    Renders an Anti-Gravity 3D floating network graph using Plotly Scatter3d with custom camera presets.
    """
    if not nodes_dict_list:
        fig = go.Figure()
        fig.update_layout(title="No Graph Data Available", template="plotly_dark")
        return fig

    # Build NetworkX graph locally for 3D layout math
    G_simple = nx.Graph()
    for n in nodes_dict_list:
        G_simple.add_node(str(n["id"]))
    for e in edges_dict_list:
        src, tgt = str(e["source"]), str(e["target"])
        if G_simple.has_node(src) and G_simple.has_node(tgt):
            G_simple.add_edge(src, tgt)

    # 3D Spring Layout
    pos_3d = nx.spring_layout(G_simple, dim=3, seed=42, k=0.5)

    edge_x, edge_y, edge_z = [], [], []
    highlight_x, highlight_y, highlight_z = [], [], []
    
    path_nodes_set = set(highlight_path) if highlight_path else set()

    for e in edges_dict_list:
        u, v = str(e["source"]), str(e["target"])
        if u in pos_3d and v in pos_3d:
            x0, y0, z0 = pos_3d[u]
            x1, y1, z1 = pos_3d[v]
            
            if u in path_nodes_set and v in path_nodes_set:
                highlight_x.extend([x0, x1, None])
                highlight_y.extend([y0, y1, None])
                highlight_z.extend([z0, z1, None])
            else:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(0, 243, 255, 0.25)', width=2),
        hoverinfo='none',
        name='Network Edge'
    )

    highlight_edge_trace = go.Scatter3d(
        x=highlight_x, y=highlight_y, z=highlight_z,
        mode='lines',
        line=dict(color='#ff2a6d', width=8),
        hoverinfo='none',
        name='Suspect Connection Path'
    )

    node_x, node_y, node_z = [], [], []
    node_colors = []
    node_sizes = []
    node_text = []

    for n in nodes_dict_list:
        nid = str(n["id"])
        if nid in pos_3d:
            x, y, z = pos_3d[nid]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            etype = n.get("type", "Entity")
            risk = float(n.get("risk_score", 50))
            name = n.get("name", nid)
            role = n.get("role", "N/A")
            loc = n.get("location", "N/A")
            details = n.get("details", "")

            if color_by == "risk":
                color = get_node_color_by_risk(risk)
            else:
                color = ENTITY_COLOR_MAP.get(etype, "#00f3ff")
            
            if highlight_path and nid in path_nodes_set:
                color = "#ffffff"

            node_colors.append(color)

            size = 12 + (risk / 10.0)
            if highlight_path and nid in path_nodes_set:
                size += 8
            node_sizes.append(size)

            hover_html = (
                f"<b>{name}</b> ({nid})<br>"
                f"Type: {etype}<br>"
                f"Threat Score: <b style='color:{color}'>{risk}/100</b><br>"
                f"Role: {role}<br>"
                f"Location: {loc}<br>"
                f"<i>{details}</i>"
            )
            node_text.append(hover_html)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=node_sizes,
            color=node_colors,
            line=dict(color='rgba(255, 255, 255, 0.8)', width=1),
            opacity=0.9
        ),
        text=[n.get("name", str(n["id"])) if float(n.get("risk_score", 0)) >= 80 else "" for n in nodes_dict_list],
        textposition="top center",
        textfont=dict(color="#00f3ff", size=10, family="Orbitron"),
        hoverinfo='text',
        hovertext=node_text,
        name='Criminal Entity'
    )

    fig = go.Figure(data=[edge_trace, highlight_edge_trace, node_trace])
    cam = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["Isometric"])

    fig.update_layout(
        paper_bgcolor='#050811',
        plot_bgcolor='#050811',
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='#050811'),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='#050811'),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor='#050811'),
            camera=cam
        ),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="#00f3ff",
            font=dict(family="Rajdhani", size=14, color="#f8fafc")
        )
    )

    return fig

@st.cache_data
def generate_2d_network_plot(nodes_dict_list, edges_dict_list, color_by="type", highlight_path=None):
    """Renders a Tactical 2D Network Grid using Plotly Scatter."""
    if not nodes_dict_list:
        return go.Figure()

    G_simple = nx.Graph()
    for n in nodes_dict_list:
        G_simple.add_node(str(n["id"]))
    for e in edges_dict_list:
        src, tgt = str(e["source"]), str(e["target"])
        if G_simple.has_node(src) and G_simple.has_node(tgt):
            G_simple.add_edge(src, tgt)

    pos_2d = nx.spring_layout(G_simple, seed=42)

    edge_x, edge_y = [], []
    for e in edges_dict_list:
        u, v = str(e["source"]), str(e["target"])
        if u in pos_2d and v in pos_2d:
            x0, y0 = pos_2d[u]
            x1, y1 = pos_2d[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(0, 243, 255, 0.3)'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, colors, sizes, text = [], [], [], [], []
    path_nodes_set = set(highlight_path) if highlight_path else set()

    for n in nodes_dict_list:
        nid = str(n["id"])
        if nid in pos_2d:
            x, y = pos_2d[nid]
            node_x.append(x)
            node_y.append(y)
            
            etype = n.get("type", "Entity")
            risk = float(n.get("risk_score", 50))
            name = n.get("name", nid)

            c = get_node_color_by_risk(risk) if color_by == "risk" else ENTITY_COLOR_MAP.get(etype, "#00f3ff")
            if highlight_path and nid in path_nodes_set:
                c = "#ffffff"
            colors.append(c)
            sizes.append(14 + (risk / 8.0))
            text.append(f"{name} ({etype}) - Risk: {risk}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n.get("name", str(n["id"])) for n in nodes_dict_list],
        textposition="top center",
        textfont=dict(color="#00f3ff", size=10, family="Orbitron"),
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color='white', width=1)
        ),
        hoverinfo='text',
        hovertext=text
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor='#050811',
        plot_bgcolor='#050811',
        margin=dict(l=0, r=0, b=0, t=20),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def generate_threat_gauge_chart(score):
    """Renders a high-tech sci-fi speedometer risk index gauge."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "NETWORK THREAT INDEX", 'font': {'size': 14, 'color': "#00f3ff", 'family': "Orbitron"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#00f3ff"},
            'bar': {'color': "#ff2a6d" if score >= 75 else "#00f3ff"},
            'bgcolor': "rgba(15, 23, 42, 0.8)",
            'borderwidth': 2,
            'bordercolor': "#00f3ff",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 255, 102, 0.15)'},
                {'range': [50, 75], 'color': 'rgba(255, 170, 0, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(255, 42, 109, 0.25)'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="Share Tech Mono"),
        margin=dict(l=30, r=30, t=40, b=20),
        height=220
    )
    return fig

def generate_entity_breakdown_chart(nodes_df):
    if nodes_df.empty or "type" not in nodes_df.columns:
        return go.Figure()
        
    counts = nodes_df["type"].value_counts().reset_index()
    counts.columns = ["Entity Type", "Count"]

    fig = px.bar(
        counts,
        x="Entity Type",
        y="Count",
        color="Entity Type",
        color_discrete_map=ENTITY_COLOR_MAP,
        title="CRIMINAL ENTITY BREAKDOWN BY CATEGORY"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="Orbitron"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def generate_risk_distribution_chart(nodes_df):
    if nodes_df.empty or "risk_score" not in nodes_df.columns:
        return go.Figure()
        
    def get_risk_cat(score):
        if score >= 85: return "Critical (85-100)"
        elif score >= 70: return "High (70-84)"
        elif score >= 50: return "Medium (50-69)"
        else: return "Low (0-49)"

    nodes_df_cat = nodes_df.copy()
    nodes_df_cat["Risk Level"] = nodes_df_cat["risk_score"].apply(get_risk_cat)
    counts = nodes_df_cat["Risk Level"].value_counts().reset_index()
    counts.columns = ["Risk Level", "Count"]

    fig = px.pie(
        counts,
        names="Risk Level",
        values="Count",
        color="Risk Level",
        color_discrete_map=RISK_COLOR_MAP,
        hole=0.5,
        title="THREAT RISK ASSESSMENT MATRIX"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc", family="Orbitron"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
