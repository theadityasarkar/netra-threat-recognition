import json
import re
import pandas as pd
import streamlit as st

@st.cache_data
def load_sample_dataset(file_path: str = "data/sample_crime_data.json"):
    """Loads the pre-packaged sample criminal network dataset."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        nodes_df = pd.DataFrame(data.get("nodes", []))
        edges_df = pd.DataFrame(data.get("edges", []))
        return nodes_df, edges_df, data
    except Exception as e:
        st.error(f"Error loading sample dataset: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data
def parse_uploaded_data(uploaded_file):
    """
    Parses CSV or JSON uploaded data into standard nodes and edges DataFrames.
    """
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".json"):
            data = json.load(uploaded_file)
            nodes_df = pd.DataFrame(data.get("nodes", []))
            edges_df = pd.DataFrame(data.get("edges", []))
            return nodes_df, edges_df, data
        
        elif filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            # If CSV contains source/target, treat as edge list
            if "source" in df.columns and "target" in df.columns:
                edges_df = df.copy()
                if "type" not in edges_df.columns:
                    edges_df["type"] = "LINKED_TO"
                if "weight" not in edges_df.columns:
                    edges_df["weight"] = 1
                
                # Derive unique nodes
                node_ids = list(set(edges_df["source"]).union(set(edges_df["target"])))
                nodes_data = []
                for nid in node_ids:
                    nid_str = str(nid)
                    nodes_data.append({
                        "id": nid_str,
                        "name": nid_str,
                        "type": "Person" if "P-" in nid_str or "Accused" in nid_str else "Entity",
                        "risk_score": 60,
                        "role": "Suspect",
                        "location": "Unknown",
                        "details": "Imported via CSV record"
                    })
                nodes_df = pd.DataFrame(nodes_data)
                return nodes_df, edges_df, {"nodes": nodes_data, "edges": edges_df.to_dict("records")}
            else:
                # Assume entity list
                nodes_df = df.copy()
                if "id" not in nodes_df.columns:
                    nodes_df["id"] = [f"N-{i+1}" for i in range(len(nodes_df))]
                edges_df = pd.DataFrame(columns=["source", "target", "type", "weight", "timestamp", "details"])
                return nodes_df, edges_df, {"nodes": nodes_df.to_dict("records"), "edges": []}
    except Exception as e:
        st.error(f"Failed to parse file: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), {}

def extract_entities_from_text(raw_text: str):
    """
    NLP & Pattern Extraction engine for unstructured FIRs, police reports, and CDR text snippets.
    Returns extracted nodes and generated relationship edges.
    """
    nodes = []
    edges = []
    
    # 1. Regex Entity Matching
    phone_pattern = r'(\+91[\-\s]?\d{10}|\b\d{10}\b)'
    vehicle_pattern = r'\b[A-Z]{2}[\-\s]?\d{2}[\-\s]?[A-Z]{1,2}[\-\s]?\d{4}\b'
    bank_pattern = r'\b(HDFC|ICICI|SBI|AXIS|PNB|BOB)[\-a-zA-Z0-9]{5,15}\b'
    fir_pattern = r'\bFIR\s*#?\s*\d+\/\d{4}\b'
    
    phones = list(set(re.findall(phone_pattern, raw_text)))
    vehicles = list(set(re.findall(vehicle_pattern, raw_text, re.IGNORECASE)))
    bank_accs = list(set([match[0] + match[1] if isinstance(match, tuple) else match for match in re.findall(bank_pattern, raw_text)]))
    firs = list(set(re.findall(fir_pattern, raw_text, re.IGNORECASE)))
    
    # Simple Named Entity fallback for Persons & Locations
    person_names = list(set(re.findall(r'\b(?:Mr\.|Mrs\.|Accused|Suspect|Alias)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', raw_text)))
    locations = list(set(re.findall(r'\b(Delhi|Mumbai|Gurugram|Noida|Jaipur|Kolkata|Bengaluru|Chennai|Hyderabad|Meerut|Ahmedabad)\b', raw_text, re.IGNORECASE)))
    
    # Build Suspect Nodes
    suspect_id = "EXT-P1"
    if person_names:
        for idx, name in enumerate(person_names[:5]):
            pid = f"EXT-P{idx+1}"
            nodes.append({
                "id": pid,
                "name": name,
                "type": "Person",
                "risk_score": 75 + (idx * 3),
                "role": "Extracted Suspect",
                "location": locations[0] if locations else "NCR Region",
                "details": f"Extracted from unstructured FIR log text."
            })
    else:
        # Default fallback suspect
        nodes.append({
            "id": suspect_id,
            "name": "Unidentified Suspect A1",
            "type": "Person",
            "risk_score": 80,
            "role": "Primary Subject",
            "location": locations[0] if locations else "Delhi NCR",
            "details": "Main suspect mentioned in uploaded document."
        })

    primary_pid = nodes[0]["id"]

    # Build Phone Nodes & Edges
    for idx, phone in enumerate(phones):
        phone_id = f"EXT-CDR-{idx+1}"
        nodes.append({
            "id": phone_id,
            "name": phone if isinstance(phone, str) else phone[0],
            "type": "Phone",
            "risk_score": 70,
            "role": "Extracted CDR Line",
            "location": "Telecom Cell",
            "details": "Extracted call record line."
        })
        edges.append({
            "source": primary_pid,
            "target": phone_id,
            "type": "USES_PHONE",
            "weight": 8,
            "timestamp": "2026-03-01",
            "details": "CDRs linked in text analysis"
        })

    # Build Vehicle Nodes & Edges
    for idx, veh in enumerate(vehicles):
        veh_id = f"EXT-VEH-{idx+1}"
        nodes.append({
            "id": veh_id,
            "name": veh.upper(),
            "type": "Vehicle",
            "risk_score": 68,
            "role": "Suspect Transport Vehicle",
            "location": "ANPR Record",
            "details": "Extracted vehicle registration."
        })
        edges.append({
            "source": primary_pid,
            "target": veh_id,
            "type": "USED_VEHICLE",
            "weight": 7,
            "timestamp": "2026-03-01",
            "details": "ANPR toll plate match"
        })

    # Build Bank Nodes & Edges
    for idx, bnk in enumerate(bank_accs):
        bnk_id = f"EXT-BNK-{idx+1}"
        nodes.append({
            "id": bnk_id,
            "name": str(bnk),
            "type": "Bank Account",
            "risk_score": 82,
            "role": "Extracted Mule Account",
            "location": "Banking Switch",
            "details": "Identified financial node."
        })
        edges.append({
            "source": primary_pid,
            "target": bnk_id,
            "type": "MONEY_TRANSFER",
            "weight": 9,
            "timestamp": "2026-03-01",
            "details": "Suspicious transaction link"
        })

    # Build FIR Nodes & Edges
    for idx, fir in enumerate(firs):
        fir_id = f"EXT-FIR-{idx+1}"
        nodes.append({
            "id": fir_id,
            "name": str(fir),
            "type": "FIR Case File",
            "risk_score": 90,
            "role": "Extracted FIR Reference",
            "location": "NCRB Division",
            "details": "Law enforcement record reference."
        })
        edges.append({
            "source": primary_pid,
            "target": fir_id,
            "type": "MENTIONED_IN_FIR",
            "weight": 10,
            "timestamp": "2026-03-01",
            "details": "Direct mention in FIR description"
        })

    return pd.DataFrame(nodes), pd.DataFrame(edges)
