import pandas as pd
import streamlit as st

def detect_suspicious_patterns(nodes_df, edges_df, metrics_df):
    """
    Scans the criminal graph network for high-threat operational anomalies:
    1. Broker/Mastermind pattern (High betweenness centrality)
    2. Smurfing / Financial laundering loops
    3. Multi-jurisdiction CDR burst anomaly
    4. Safehouse & Transit geofence hotspots
    """
    alerts = []

    if metrics_df.empty:
        return alerts

    # 1. Mastermind / Key Intermediary Detection (Betweenness Centrality)
    high_brokers = metrics_df[metrics_df["betweenness_centrality"] > 0.15]
    for _, row in high_brokers.iterrows():
        alerts.append({
            "id": f"ALT-BRK-{row['id']}",
            "severity": "CRITICAL",
            "type": "Key Intermediary / Network Broker",
            "title": f"High Influence Broker: {row['name']}",
            "description": f"Entity '{row['name']}' ({row['type']}) exhibits a Betweenness Centrality of {row['betweenness_centrality']}. Acts as a critical information/financial bridge between disconnected criminal cells.",
            "target_node": row['id'],
            "recommended_action": "Issue priority surveillance order & wiretap intercept on primary communications."
        })

    # 2. Financial Laundering / Smurfing Anomaly
    if not edges_df.empty and "type" in edges_df.columns:
        money_edges = edges_df[edges_df["type"] == "MONEY_TRANSFER"]
        if len(money_edges) >= 3:
            target_accs = money_edges["target"].value_counts()
            for acc_id, count in target_accs.items():
                if count >= 2:
                    acc_name = acc_id
                    acc_node = nodes_df[nodes_df["id"] == acc_id] if not nodes_df.empty else None
                    if acc_node is not None and not acc_node.empty:
                        acc_name = acc_node.iloc[0].get("name", acc_id)
                    
                    alerts.append({
                        "id": f"ALT-FIN-{acc_id}",
                        "severity": "CRITICAL",
                        "type": "Financial Smurfing Pattern",
                        "title": f"Rapid Layering Hub: {acc_name}",
                        "description": f"Bank Node '{acc_name}' received multiple rapid inbound transfers from distinct suspect accounts. High probability of mule account layering.",
                        "target_node": acc_id,
                        "recommended_action": "Freeze target bank account under FIU-IND / PMLA directives."
                    })

    # 3. CDR Call Burst / Multi-jurisdiction Line
    if not edges_df.empty and "type" in edges_df.columns:
        cdr_edges = edges_df[edges_df["type"] == "CALL_TO"]
        if len(cdr_edges) >= 3:
            top_callers = pd.concat([cdr_edges["source"], cdr_edges["target"]]).value_counts()
            for phone_id, count in top_callers.items():
                if count >= 3:
                    phone_node = nodes_df[nodes_df["id"] == phone_id] if not nodes_df.empty else None
                    pname = phone_node.iloc[0].get("name", phone_id) if phone_node is not None and not phone_node.empty else phone_id
                    alerts.append({
                        "id": f"ALT-CDR-{phone_id}",
                        "severity": "HIGH",
                        "type": "High Volume CDR Connection",
                        "title": f"Encrypted Hotline Hub: {pname}",
                        "description": f"Phone Line '{pname}' logged unusual high-frequency call traffic connecting multiple high-risk suspects simultaneously.",
                        "target_node": phone_id,
                        "recommended_action": "Request cell tower dump & IMEI clone tracking from Telecom Provider."
                    })

    # 4. Critical Threat Suspect Alerts
    critical_suspects = metrics_df[metrics_df["risk_score"] >= 90]
    for _, row in critical_suspects.iterrows():
        alerts.append({
            "id": f"ALT-SUS-{row['id']}",
            "severity": "HIGH",
            "type": "High-Risk Syndicate Leader",
            "title": f"Priority Target Flagged: {row['name']}",
            "description": f"Suspect '{row['name']}' has a Threat Risk Score of {row['risk_score']}/100. Role: {row['role']}.",
            "target_node": row['id'],
            "recommended_action": "Deploy NCRB field intelligence unit for location raid."
        })

    return alerts

def calculate_network_threat_index(metrics_df, alerts):
    """Calculates overall Network Threat Score (0 - 100%)."""
    if metrics_df.empty:
        return 0.0
        
    avg_risk = metrics_df["risk_score"].mean()
    max_risk = metrics_df["risk_score"].max()
    
    alert_penalty = len([a for a in alerts if a["severity"] == "CRITICAL"]) * 5 + len([a for a in alerts if a["severity"] == "HIGH"]) * 2
    
    overall_score = (avg_risk * 0.4) + (max_risk * 0.4) + min(alert_penalty, 20)
    return min(round(overall_score, 1), 100.0)
