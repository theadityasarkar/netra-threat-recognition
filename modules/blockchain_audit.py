import hashlib
import time
import pandas as pd
import streamlit as st

def compute_file_hash(file_bytes: bytes) -> str:
    """Computes SHA-256 cryptographic hash of evidence binary data."""
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()

def init_blockchain_ledger():
    """Initializes session state blockchain ledger if not already initialized."""
    if "blockchain_ledger" not in st.session_state:
        # Genesis Block
        genesis_block = {
            "block_index": 0,
            "timestamp": "2026-08-20 10:00:00 UTC",
            "file_name": "GENESIS_NCRB_AUDIT_BLOCK",
            "evidence_hash": "0000000000000000000000000000000000000000000000000000000000000000",
            "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
            "investigator_id": "NCRB-SYS-ADMIN",
            "action": "LEDGER_INITIALIZED"
        }
        st.session_state.blockchain_ledger = [genesis_block]

def add_evidence_to_ledger(file_name: str, file_bytes: bytes, investigator_id: str = "INV-NCRB-9902", action: str = "EVIDENCE_INGESTED"):
    """Adds a new evidence record block to the immutable ledger."""
    init_blockchain_ledger()
    ledger = st.session_state.blockchain_ledger
    
    prev_block = ledger[-1]
    prev_hash = prev_block["evidence_hash"]
    
    evidence_hash = compute_file_hash(file_bytes)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    new_block = {
        "block_index": len(ledger),
        "timestamp": timestamp,
        "file_name": file_name,
        "evidence_hash": evidence_hash,
        "previous_hash": prev_hash,
        "investigator_id": investigator_id,
        "action": action
    }
    
    ledger.append(new_block)
    st.session_state.blockchain_ledger = ledger
    return new_block

def get_blockchain_dataframe():
    """Returns the immutable blockchain log as a formatted DataFrame."""
    init_blockchain_ledger()
    return pd.DataFrame(st.session_state.blockchain_ledger)

def verify_evidence_hash(file_bytes: bytes, target_hash: str) -> bool:
    """Verifies whether an evidence file matches its recorded SHA-256 hash signature."""
    current_hash = compute_file_hash(file_bytes)
    return current_hash.lower() == target_hash.lower()
