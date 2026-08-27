"""
Layer 1 — synthetic criminal-network generator.

Produces a realistic, *entirely fictional* dataset so the dashboard is demo-ready with
zero setup. The generator is deterministic for a given seed, and deliberately plants
structures that the analysis layer is supposed to find:

  • hierarchical syndicates          → Kingpin / Lieutenant / Foot-Soldier roles
  • sparse inter-syndicate brokers   → high betweenness, low degree
  • a financial hub with no priors   → money-mule rule
  • an isolated high-severity actor  → structurally isolated offender rule
  • a dormant-then-reactivated actor → dormancy-spike rule

It also emits a corpus of narrative intelligence documents built from the same
actors and cases, so the entity-extraction layer has something real to work on
(see :func:`_build_intel_documents`).

Nothing here corresponds to a real person, FIR, or organisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CRIME_SEVERITY,
    CRIME_TYPES,
    DOCUMENT_COLUMNS,
    EVIDENCE_SOURCES,
    PERSON_STATUS,
    RELATION_TYPES,
    RISK_CATEGORIES,
    WOMEN_SAFETY_CRIMES,
)

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

# ──────────────────────────────────────────────────────────────────────────────
# Fictional name pools (procedurally combined — not real identities)
# ──────────────────────────────────────────────────────────────────────────────

_FIRST = [
    "Aarav", "Vikram", "Rohit", "Imran", "Sameer", "Kabir", "Nitin", "Farhan", "Devendra",
    "Suresh", "Manoj", "Rakesh", "Zahid", "Prakash", "Anil", "Yusuf", "Tarun", "Gopal",
    "Harish", "Naveen", "Salim", "Dinesh", "Ravi", "Ashok", "Javed", "Mahesh", "Pankaj",
    "Bhupinder", "Krishna", "Altaf", "Sunil", "Deepak", "Rajesh", "Firoz", "Mukesh",
    "Meera", "Anjali", "Zoya", "Kavita", "Sunita", "Rehana", "Pooja", "Nisha", "Shalini",
    "Farida", "Rukmini", "Sameena", "Lata", "Bhavna",
]

_LAST = [
    "Sharma", "Verma", "Khan", "Reddy", "Patel", "Nair", "Chauhan", "Yadav", "Singh",
    "Bose", "Mehta", "Iyer", "Joshi", "Kulkarni", "Malhotra", "Ansari", "Pillai", "Rathore",
    "Gowda", "Dutta", "Mishra", "Naik", "Sethi", "Bhatt", "Qureshi", "Deshmukh", "Kapoor",
    "Choudhury", "Panicker", "Thakur", "Saxena", "Barman", "Shetty", "Solanki", "Rawat",
]

_ALIAS_TOKENS = [
    "Chhota", "Bada", "Kala", "Lambu", "Tiger", "Bhai", "Ustad", "Chacha", "Doctor",
    "Guru", "Pehelwan", "Seth", "Munna", "Raja", "Kaka", "Bond", "Sultan", "Dada",
]

_OCCUPATION_COVER = [
    "Scrap Dealer", "Transport Operator", "Real Estate Agent", "Hotel Owner",
    "Mobile Recharge Shop", "Travel Agent", "Import-Export Trader", "Cattle Trader",
    "Construction Contractor", "Money Exchange Operator", "Fishing Boat Owner",
    "Auto Driver", "Unemployed", "Cyber Cafe Owner", "Jewellery Shop Assistant",
    "Security Guard", "Pharmacy Assistant", "Truck Driver", "Chartered Accountant",
]

# (city, state) pairs — chosen to give the geographic panel realistic spread.
_LOCATIONS = [
    ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Nagpur", "Maharashtra"),
    ("Delhi", "Delhi"), ("Gurugram", "Haryana"), ("Ludhiana", "Punjab"),
    ("Amritsar", "Punjab"), ("Jaipur", "Rajasthan"), ("Jodhpur", "Rajasthan"),
    ("Ahmedabad", "Gujarat"), ("Surat", "Gujarat"), ("Indore", "Madhya Pradesh"),
    ("Lucknow", "Uttar Pradesh"), ("Kanpur", "Uttar Pradesh"), ("Varanasi", "Uttar Pradesh"),
    ("Patna", "Bihar"), ("Kolkata", "West Bengal"), ("Siliguri", "West Bengal"),
    ("Guwahati", "Assam"), ("Ranchi", "Jharkhand"), ("Bhubaneswar", "Odisha"),
    ("Hyderabad", "Telangana"), ("Vijayawada", "Andhra Pradesh"), ("Bengaluru", "Karnataka"),
    ("Mangaluru", "Karnataka"), ("Chennai", "Tamil Nadu"), ("Coimbatore", "Tamil Nadu"),
    ("Kochi", "Kerala"), ("Thiruvananthapuram", "Kerala"), ("Srinagar", "Jammu & Kashmir"),
]

# Fictional syndicate profiles: (label, primary crime, secondary crime, home index)
_SYNDICATES = [
    ("Coastal Narcotics Module", "Narcotics Trafficking", "Money Laundering", 0),
    ("Northern Arms Corridor", "Arms Smuggling", "Counterfeit Currency", 5),
    ("Metro Cyber Fraud Ring", "Cyber Fraud", "Money Laundering", 23),
    ("Eastern Trafficking Chain", "Human Trafficking", "Kidnapping", 16),
    ("Central Extortion Syndicate", "Extortion", "Organised Robbery", 12),
    ("Hawala Settlement Network", "Hawala Operation", "Money Laundering", 9),
    ("Western Vehicle Theft Ring", "Vehicle Theft Ring", "Organised Robbery", 10),
]


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_sample_dataset(
    n_persons: int = 220,
    n_syndicates: int = 6,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build a synthetic persons / relationships / crime_records triple.

    Args:
        n_persons: total number of actors in the network (clamped to 40–2000).
        n_syndicates: number of hierarchical modules to lay down (clamped to 2–7).
        seed: RNG seed — same seed always yields the same network.

    Returns:
        ``{"persons": df, "relationships": df, "crime_records": df}``
    """
    n_persons = int(np.clip(n_persons, 40, 2000))
    n_syndicates = int(np.clip(n_syndicates, 2, len(_SYNDICATES)))
    rng = np.random.default_rng(seed)

    syndicates = _SYNDICATES[:n_syndicates]

    # ── 1. Allocate people to syndicates + a peripheral pool ──────────────────
    # ~85 % of actors belong to a module; the rest are loose peripheral contacts.
    n_affiliated = int(n_persons * 0.85)
    n_peripheral = n_persons - n_affiliated

    # Uneven module sizes look far more realistic than equal splits.
    shares = rng.dirichlet(np.linspace(1.6, 0.8, n_syndicates))
    sizes = np.maximum(6, (shares * n_affiliated).astype(int))
    # Reconcile rounding drift back into the largest module.
    sizes[int(np.argmax(sizes))] += n_affiliated - int(sizes.sum())

    people: list[dict] = []
    membership: list[list[str]] = []       # person_ids per syndicate
    roles: dict[str, str] = {}             # ground-truth seeding role (not exported)
    pid = 0

    def _make_person(module_idx: int | None, role: str) -> dict:
        nonlocal pid
        pid += 1
        person_id = f"P{pid:04d}"
        name = f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"

        # Aliases: senior actors and evaders accumulate more.
        n_alias = {"kingpin": 2, "lieutenant": 1, "financier": 1}.get(role, 0)
        if rng.random() < 0.18:
            n_alias += 1
        aliases = [
            f"{_ALIAS_TOKENS[rng.integers(len(_ALIAS_TOKENS))]} {name.split()[0]}"
            for _ in range(n_alias)
        ]

        if module_idx is None:
            loc_idx = int(rng.integers(len(_LOCATIONS)))
            primary, secondary = "Illegal Betting", "Organised Robbery"
        else:
            label, primary, secondary, home = syndicates[module_idx]
            # 70 % of a module clusters near its home city, 30 % spreads out.
            loc_idx = home if rng.random() < 0.7 else int(rng.integers(len(_LOCATIONS)))
            loc_idx = (loc_idx + int(rng.integers(0, 3))) % len(_LOCATIONS)

        city, state = _LOCATIONS[loc_idx]

        # Prior offences scale with seniority.
        base_priors = {
            "kingpin": (6, 14), "lieutenant": (4, 9), "enforcer": (3, 8),
            "financier": (1, 4), "courier": (1, 5), "soldier": (0, 4),
            "broker": (2, 6), "peripheral": (0, 2),
        }[role]
        priors = int(rng.integers(base_priors[0], base_priors[1] + 1))

        offence_pool = [primary]
        if rng.random() < 0.55:
            offence_pool.append(secondary)
        if rng.random() < 0.20:
            offence_pool.append(CRIME_TYPES[int(rng.integers(len(CRIME_TYPES)))])
        offences = sorted(set(offence_pool))

        first_arrest = int(rng.integers(2005, 2024))
        # Most actors are recently active; a slice went quiet years ago.
        if rng.random() < 0.12:
            last_year = int(rng.integers(2018, 2022))
        else:
            last_year = int(rng.integers(2024, 2027))
        last_year = max(last_year, first_arrest)
        last_activity = f"{last_year}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}"

        # Risk band follows seniority with deliberate noise (real records are messy).
        risk_bias = {
            "kingpin": 0, "lieutenant": 0, "enforcer": 1, "financier": 1,
            "broker": 1, "courier": 2, "soldier": 2, "peripheral": 3,
        }[role]
        risk_idx = int(np.clip(risk_bias + rng.integers(-1, 2), 0, len(RISK_CATEGORIES) - 1))

        status_w = np.array([0.34, 0.20, 0.16, 0.12, 0.15, 0.03])
        status = str(rng.choice(PERSON_STATUS, p=status_w / status_w.sum()))

        age = int(np.clip(
            {"kingpin": 48, "lieutenant": 40, "financier": 45, "enforcer": 34,
             "broker": 42, "courier": 29, "soldier": 27, "peripheral": 33}[role]
            + rng.integers(-7, 8), 18, 74))

        return {
            "person_id": person_id,
            "name": name,
            "alias": "; ".join(aliases),
            "age": age,
            "gender": "M" if rng.random() < 0.82 else "F",
            "city": city,
            "state": state,
            "occupation": (
                "Money Exchange Operator" if role == "financier"
                else _OCCUPATION_COVER[int(rng.integers(len(_OCCUPATION_COVER)))]
            ),
            "prior_offences": priors,
            "offence_types": "; ".join(offences),
            "first_arrest_year": first_arrest,
            "last_activity": last_activity,
            "status": status,
            "risk_category": RISK_CATEGORIES[risk_idx],
        }

    # Build each syndicate as a small hierarchy.
    for m_idx, size in enumerate(sizes):
        members: list[str] = []
        composition = ["kingpin", "financier"]
        n_lieut = max(2, int(size * 0.14))
        n_enf = max(1, int(size * 0.12))
        n_cour = max(1, int(size * 0.18))
        composition += ["lieutenant"] * n_lieut + ["enforcer"] * n_enf + ["courier"] * n_cour
        composition += ["soldier"] * max(0, int(size) - len(composition))

        for role in composition[:int(size)]:
            rec = _make_person(m_idx, role)
            people.append(rec)
            members.append(rec["person_id"])
            roles[rec["person_id"]] = role
        membership.append(members)

    for _ in range(n_peripheral):
        rec = _make_person(None, "peripheral")
        people.append(rec)
        roles[rec["person_id"]] = "peripheral"

    persons_df = pd.DataFrame(people)
    peripheral_ids = [p for p, r in roles.items() if r == "peripheral"]

    # ── 2. Edges ──────────────────────────────────────────────────────────────
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add_edge(a: str, b: str, rel: str, strength: float, interactions: int | None = None):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        first_y = int(rng.integers(2016, 2025))
        last_y = int(rng.integers(first_y, 2027))
        edges.append({
            "source_id": key[0],
            "target_id": key[1],
            "relation_type": rel,
            "strength": round(float(np.clip(strength + rng.normal(0, 0.07), 0.08, 1.0)), 3),
            "interaction_count": int(interactions if interactions is not None
                                     else np.clip(rng.poisson(14) + 1, 1, 400)),
            "first_seen": f"{first_y}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
            "last_seen": f"{last_y}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
            "evidence_source": EVIDENCE_SOURCES[int(rng.integers(len(EVIDENCE_SOURCES)))],
            "location": _LOCATIONS[int(rng.integers(len(_LOCATIONS)))][0],
        })

    for members in membership:
        by_role: dict[str, list[str]] = {}
        for p in members:
            by_role.setdefault(roles[p], []).append(p)

        kingpins = by_role.get("kingpin", [])
        lieutenants = by_role.get("lieutenant", [])
        enforcers = by_role.get("enforcer", [])
        couriers = by_role.get("courier", [])
        soldiers = by_role.get("soldier", [])
        financiers = by_role.get("financier", [])

        # Kingpin ↔ lieutenants / financier: strong, low-volume, insulated links.
        for lt in lieutenants:
            _add_edge(kingpins[0], lt, "Gang Member", 0.90, int(rng.integers(4, 20)))
        for fin in financiers:
            _add_edge(kingpins[0], fin, "Financial", 0.88, int(rng.integers(3, 15)))
            for lt in lieutenants:
                if rng.random() < 0.5:
                    _add_edge(fin, lt, "Financial", 0.70)

        # Lieutenants command enforcers, couriers and soldiers.
        subordinates = enforcers + couriers + soldiers
        rng.shuffle(subordinates)
        if lieutenants:
            for i, sub in enumerate(subordinates):
                handler = lieutenants[i % len(lieutenants)]
                rel = "Handler" if roles[sub] == "courier" else "Gang Member"
                _add_edge(handler, sub, rel, 0.72)
                # A second handler for redundancy in ~20 % of cases.
                if rng.random() < 0.20 and len(lieutenants) > 1:
                    alt = lieutenants[(i + 1) % len(lieutenants)]
                    _add_edge(alt, sub, "Communication", 0.45)

        # Peer chatter inside the module (this is what creates realistic density).
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if rng.random() < 0.055:
                    rel = str(rng.choice(
                        ["Communication", "Associate", "Co-Accused", "Family"],
                        p=[0.45, 0.30, 0.18, 0.07]))
                    _add_edge(a, b, rel, 0.40)

        # Couriers ferry between soldiers.
        for c in couriers:
            for s in rng.choice(soldiers, size=min(3, len(soldiers)), replace=False) if soldiers else []:
                _add_edge(c, str(s), "Courier Link", 0.60)

        # Peripheral contacts hang off the module edge (shelter, family, betting).
        if peripheral_ids and soldiers:
            for _ in range(max(1, len(members) // 10)):
                p = str(rng.choice(peripheral_ids))
                s = str(rng.choice(soldiers))
                _add_edge(p, s, str(rng.choice(["Shelter Provider", "Family", "Associate"])), 0.30)

    # ── 3. Planted anomalies ──────────────────────────────────────────────────
    planted: dict[str, list[str]] = {}

    # (a) Brokers: sparse bridges between modules — high betweenness, low degree.
    #     These are the actors a degree-only analysis would completely miss. Pair
    #     well-separated syndicates (0↔2, 1↔3, …) rather than adjacent ones, so the
    #     two halves rarely fall in the same detected module — an adjacent pair that
    #     Louvain merges would leave the broker with no cross-module signal at all.
    broker_ids: list[str] = []
    gap = max(2, n_syndicates // 2)
    # Three brokers is enough to demonstrate the pattern. Planting one per module pair
    # floods the graph with cross-module paths, which destroys the very betweenness
    # signal that makes a broker detectable.
    broker_pairs = [(i, (i + gap) % n_syndicates) for i in range(min(3, n_syndicates - 1))]
    broker_pairs = [(a, b) for a, b in broker_pairs if a != b]
    for a_mod, b_mod in broker_pairs:
        pid += 1
        broker_id = f"P{pid:04d}"
        name = f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"
        persons_df.loc[len(persons_df)] = {
            "person_id": broker_id,
            "name": name,
            "alias": f"{_ALIAS_TOKENS[rng.integers(len(_ALIAS_TOKENS))]} {name.split()[0]}",
            "age": int(rng.integers(35, 55)),
            "gender": "M" if rng.random() < 0.8 else "F",
            "city": _LOCATIONS[int(rng.integers(len(_LOCATIONS)))][0],
            "state": _LOCATIONS[int(rng.integers(len(_LOCATIONS)))][1],
            "occupation": "Transport Operator",
            "prior_offences": int(rng.integers(1, 4)),
            "offence_types": "Narcotics Trafficking; Money Laundering",
            "first_arrest_year": int(rng.integers(2012, 2022)),
            "last_activity": f"2026-{rng.integers(1, 9):02d}-{rng.integers(1, 29):02d}",
            "status": "Under Surveillance",
            "risk_category": "Medium",
        }
        roles[broker_id] = "broker"
        broker_ids.append(broker_id)

        # One senior contact in each of two well-separated modules.
        for m in (a_mod, b_mod):
            pool = [p for p in membership[m] if roles[p] in ("lieutenant", "kingpin")]
            if pool:
                _add_edge(broker_id, str(rng.choice(pool)), "Financial", 0.80,
                          int(rng.integers(2, 8)))
    planted["cross_module_broker"] = broker_ids

    # (a2) Gateway chokepoint: a small satellite cell joined to the rest of the network
    #      through EXACTLY ONE liaison. Removing the liaison isolates the entire cell —
    #      a genuine articulation point, and the clearest possible answer to "who do we
    #      arrest first to disrupt the network?".
    gateway_id = f"P{pid + 1:04d}"
    cell_size = 7
    cell_ids: list[str] = []
    for j in range(cell_size + 1):
        pid += 1
        cid = f"P{pid:04d}"
        cname = f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"
        is_gateway = (cid == gateway_id)
        persons_df.loc[len(persons_df)] = {
            "person_id": cid, "name": cname,
            "alias": (f"Seth {cname.split()[0]}; Doctor {cname.split()[1]}" if is_gateway else ""),
            "age": int(rng.integers(38, 58)) if is_gateway else int(rng.integers(22, 40)),
            "gender": "M" if rng.random() < 0.85 else "F",
            "city": "Siliguri", "state": "West Bengal",
            "occupation": "Import-Export Trader" if is_gateway else _OCCUPATION_COVER[
                int(rng.integers(len(_OCCUPATION_COVER)))],
            "prior_offences": int(rng.integers(4, 9)) if is_gateway else int(rng.integers(0, 4)),
            "offence_types": "Human Trafficking; Arms Smuggling" if is_gateway
                             else "Human Trafficking",
            "first_arrest_year": int(rng.integers(2010, 2020)),
            "last_activity": f"2026-0{rng.integers(1, 9)}-{rng.integers(10, 28)}",
            "status": "Under Surveillance" if is_gateway else "Active",
            "risk_category": "Critical" if is_gateway else "Medium",
        }
        roles[cid] = "lieutenant" if is_gateway else "soldier"
        cell_ids.append(cid)

    # Internal cell structure (dense-ish), then a single external link via the gateway.
    for i, ca in enumerate(cell_ids):
        for cb in cell_ids[i + 1:]:
            if ca == gateway_id or cb == gateway_id or rng.random() < 0.5:
                _add_edge(ca, cb, str(rng.choice(["Gang Member", "Co-Accused", "Communication"])),
                          0.6)
    # The ONLY bridge from the cell to the wider network:
    external_anchor = [p for p in membership[3 % n_syndicates]
                       if roles[p] in ("lieutenant", "kingpin")]
    if external_anchor:
        _add_edge(gateway_id, str(rng.choice(external_anchor)), "Handler", 0.85,
                  int(rng.integers(4, 12)))
    planted["gateway_chokepoint"] = [gateway_id]

    # (b) Money mule: many Financial links, almost no criminal record.
    pid += 1
    mule_id = f"P{pid:04d}"
    mule_name = f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"
    persons_df.loc[len(persons_df)] = {
        "person_id": mule_id, "name": mule_name, "alias": "",
        "age": int(rng.integers(24, 34)), "gender": "F" if rng.random() < 0.5 else "M",
        "city": "Surat", "state": "Gujarat", "occupation": "Chartered Accountant",
        "prior_offences": 0, "offence_types": "", "first_arrest_year": 2025,
        "last_activity": "2026-07-19", "status": "Active", "risk_category": "Low",
    }
    roles[mule_id] = "financier"
    fin_targets = [p for m in membership for p in m if roles[p] in ("financier", "kingpin")]
    for t in rng.choice(fin_targets, size=min(9, len(fin_targets)), replace=False):
        _add_edge(mule_id, str(t), "Financial", 0.66, int(rng.integers(20, 90)))
    planted["money_mule"] = [mule_id]

    # (c) Isolated high-severity actor: barely connected, very serious offences.
    pid += 1
    lone_id = f"P{pid:04d}"
    lone_name = f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"
    persons_df.loc[len(persons_df)] = {
        "person_id": lone_id, "name": lone_name,
        "alias": f"Ustad {lone_name.split()[0]}; Doctor {lone_name.split()[1]}; Bond",
        "age": int(rng.integers(40, 60)), "gender": "M",
        "city": "Srinagar", "state": "Jammu & Kashmir", "occupation": "Import-Export Trader",
        "prior_offences": 9, "offence_types": "Arms Smuggling; Counterfeit Currency",
        "first_arrest_year": 2011, "last_activity": "2026-06-02",
        "status": "Absconding", "risk_category": "Critical",
    }
    roles[lone_id] = "kingpin"
    lone_contact = str(rng.choice(membership[1]))
    _add_edge(lone_id, lone_contact, "Handler", 0.92, 3)
    planted["isolated_high_severity"] = [lone_id]

    # (d) Dormancy spike: quiet for years, then a burst of fresh contacts.
    pid += 1
    dorm_id = f"P{pid:04d}"
    dorm_name = f"{_FIRST[rng.integers(len(_FIRST))]} {_LAST[rng.integers(len(_LAST))]}"
    persons_df.loc[len(persons_df)] = {
        "person_id": dorm_id, "name": dorm_name, "alias": f"Kaka {dorm_name.split()[0]}",
        "age": int(rng.integers(45, 62)), "gender": "M",
        "city": "Ludhiana", "state": "Punjab", "occupation": "Scrap Dealer",
        "prior_offences": 7, "offence_types": "Arms Smuggling; Extortion",
        "first_arrest_year": 2009, "last_activity": "2026-08-11",
        "status": "Active", "risk_category": "High",
    }
    roles[dorm_id] = "lieutenant"
    burst_pool = [p for m in membership[:2] for p in m]
    for t in rng.choice(burst_pool, size=min(7, len(burst_pool)), replace=False):
        key = (dorm_id, str(t)) if dorm_id < str(t) else (str(t), dorm_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source_id": key[0], "target_id": key[1], "relation_type": "Communication",
            "strength": round(float(rng.uniform(0.5, 0.8)), 3),
            "interaction_count": int(rng.integers(30, 120)),
            "first_seen": "2026-04-02", "last_seen": "2026-08-15",
            "evidence_source": "CDR Analysis", "location": "Ludhiana",
        })
    planted["dormancy_spike"] = [dorm_id]

    # ── Connectivity guarantee ────────────────────────────────────────────────
    # Attach any actor that ended up with no links to a random module member, so the
    # network has no isolated singletons. Isolated nodes are not interesting network
    # intelligence — they only generate statistical false positives downstream.
    connected: set[str] = set()
    for e in edges:
        connected.add(e["source_id"])
        connected.add(e["target_id"])
    all_ids = list(persons_df["person_id"])
    anchor_pool = [pid for m in membership for pid in m
                   if roles.get(pid) in ("soldier", "courier", "lieutenant")] or all_ids
    for pid in all_ids:
        if pid in connected:
            continue
        anchor = str(rng.choice(anchor_pool))
        _add_edge(pid, anchor, str(rng.choice(["Associate", "Family", "Shelter Provider"])), 0.28)
        connected.add(pid)

    relationships_df = pd.DataFrame(edges)

    # ── 4. Crime records (FIRs) ───────────────────────────────────────────────
    all_members = [m for m in membership]
    crimes: list[dict] = []
    n_cases = max(20, int(len(persons_df) * 0.45))

    for c in range(1, n_cases + 1):
        m_idx = int(rng.integers(len(all_members)))
        label, primary, secondary, _ = syndicates[m_idx]
        crime_type = primary if rng.random() < 0.6 else secondary
        if rng.random() < 0.12:
            crime_type = CRIME_TYPES[int(rng.integers(len(CRIME_TYPES)))]

        pool = all_members[m_idx]
        n_acc = int(np.clip(rng.poisson(2.4) + 1, 1, min(7, len(pool))))
        accused = list(rng.choice(pool, size=n_acc, replace=False))
        # A small share of cases are genuine joint operations across modules — the real
        # intel signal. Kept deliberately low (~6 %) so the planted brokers remain the
        # dominant inter-module bridges rather than being drowned by co-accused paths.
        if rng.random() < 0.06 and len(all_members) > 1:
            other = all_members[(m_idx + 1) % len(all_members)]
            accused += list(rng.choice(other, size=1, replace=False))

        year = int(rng.integers(2021, 2027))
        base_sev = CRIME_SEVERITY.get(crime_type, 3)
        severity = int(np.clip(base_sev + rng.integers(-1, 2), 1, 5))
        loc_idx = int(rng.integers(len(_LOCATIONS)))

        crimes.append({
            "case_id": f"C{c:05d}",
            "fir_number": f"FIR/{year}/{rng.integers(100, 999)}/{rng.integers(1, 99):02d}",
            "crime_type": crime_type,
            "date": f"{year}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
            "city": _LOCATIONS[loc_idx][0],
            "state": _LOCATIONS[loc_idx][1],
            "severity": severity,
            "status": str(rng.choice(
                ["Under Investigation", "Charge-sheeted", "Trial Ongoing", "Closed"],
                p=[0.38, 0.28, 0.22, 0.12])),
            "persons_involved": "; ".join(str(a) for a in accused),
        })

    crimes_df = pd.DataFrame(crimes)

    # ── 5. Unstructured intelligence documents ────────────────────────────────
    documents_df = _build_intel_documents(persons_df, crimes_df, rng)

    # Attach the planted-anomaly key as metadata for the docs/demo script. It is
    # informational only — no analysis module ever reads it.
    persons_df.attrs["planted_anomalies"] = planted
    persons_df.attrs["syndicates"] = [s[0] for s in syndicates]

    return {
        "persons": persons_df.reset_index(drop=True),
        "relationships": relationships_df.reset_index(drop=True),
        "crime_records": crimes_df.reset_index(drop=True),
        "documents": documents_df.reset_index(drop=True),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Unstructured intelligence documents
#
# PS 26189 names four unstructured sources — FIR narratives, surveillance reports,
# social-media intelligence and intelligence-agency reports — and requires entity
# extraction from them. The demo corpus is therefore generated *from the structured
# data*, embedding the real names, cities and case ids of actors who already exist
# in the graph. That matters: an analyst (or a judge) can extract "Krishna Bose"
# from an FIR narrative and immediately find the same actor in the network view.
# A corpus of unrelated lorem text would demonstrate the regexes and nothing else.
#
# The identifiers (phones, plates, IMEIs, accounts, wallets) are synthetic and
# drawn from the same seeded RNG, so the corpus is byte-identical run to run.
# ──────────────────────────────────────────────────────────────────────────────

# RTO letter codes by state, so a plate in a Kolkata narrative reads "WB ..".
_STATE_RTO = {
    "Maharashtra": "MH", "Delhi": "DL", "Haryana": "HR", "Punjab": "PB",
    "Rajasthan": "RJ", "Gujarat": "GJ", "Madhya Pradesh": "MP",
    "Uttar Pradesh": "UP", "Bihar": "BR", "West Bengal": "WB", "Assam": "AS",
    "Jharkhand": "JH", "Odisha": "OD", "Telangana": "TS", "Andhra Pradesh": "AP",
    "Karnataka": "KA", "Tamil Nadu": "TN", "Kerala": "KL", "Jammu & Kashmir": "JK",
}

_FRONT_BUSINESSES = [
    "Bright Future Placement Agency", "Skyline Manpower Consultancy",
    "Golden Route Travels", "Riverfront Lodge", "Sunrise Overseas Recruiters",
    "Blue Orchid Spa", "Metro Exim Traders", "Deccan Logistics Pvt Ltd",
    "Crescent Forex Exchange", "Silver Line Tours",
]

_BORDER_POINTS = ["Raxaul", "Petrapole", "Jogbani", "Sunauli", "Banbasa", "Siliguri"]


def _synth_phone(rng) -> str:
    return f"{int(rng.integers(6, 10))}{int(rng.integers(10**8, 10**9))}"


def _synth_imei(rng) -> str:
    """15-digit IMEI with a valid Luhn check digit, so the extractor scores it high."""
    body = "".join(str(int(rng.integers(0, 10))) for _ in range(14))
    total, parity = 0, (len(body) + 1) % 2
    for i, ch in enumerate(body):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return body + str((10 - total % 10) % 10)


def _synth_plate(rng, state: str) -> str:
    code = _STATE_RTO.get(state, "MH")
    series = "".join(chr(65 + int(rng.integers(0, 26))) for _ in range(2))
    return f"{code} {int(rng.integers(1, 60)):02d} {series} {int(rng.integers(1000, 10000))}"


def _synth_wallet(rng) -> str:
    return "0x" + "".join("0123456789abcdef"[int(rng.integers(0, 16))] for _ in range(40))


def _synth_account(rng) -> str:
    return str(int(rng.integers(10**13, 10**14)))


def _synth_amount(rng) -> str:
    if rng.random() < 0.5:
        return f"Rs {int(rng.integers(2, 40))},{int(rng.integers(10, 99))},000"
    return f"Rs {int(rng.integers(1, 9))}.{int(rng.integers(1, 9))} crore"


def _build_intel_documents(persons_df: pd.DataFrame,
                           crimes_df: pd.DataFrame,
                           rng) -> pd.DataFrame:
    """Render narrative documents for the highest-value cases in the dataset.

    Women Safety Division cases are chosen first — that is the division that owns
    PS 26189 — then the remaining slots are filled with the most severe cases of
    any type, so the corpus exercises weapon, contraband and hawala vocabulary too.
    """
    if crimes_df.empty or persons_df.empty:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)

    names = dict(zip(persons_df["person_id"].astype(str), persons_df["name"].astype(str)))

    ws = crimes_df[crimes_df["crime_type"].isin(WOMEN_SAFETY_CRIMES)]
    ws = ws.sort_values("severity", ascending=False).head(9)
    rest = (crimes_df[~crimes_df["case_id"].isin(ws["case_id"])]
            .sort_values("severity", ascending=False).head(6))
    selected = pd.concat([ws, rest]) if len(ws) else rest

    rows: list[dict] = []
    for n, (_, case) in enumerate(selected.iterrows(), start=1):
        accused = [a.strip() for a in str(case["persons_involved"]).split(";") if a.strip()]
        who = [names.get(a, a) for a in accused][:3]
        if not who:
            continue

        lead = who[0]
        others = who[1:]
        city, state = str(case["city"]), str(case["state"])
        ctype, sev = str(case["crime_type"]), int(case["severity"])
        case_id, fir = str(case["case_id"]), str(case["fir_number"])
        year, month, day = str(case["date"]).split("-")
        date_txt = f"{int(day)} {_MONTHS[int(month) - 1]} {year}"

        phone, imei = _synth_phone(rng), _synth_imei(rng)
        plate, wallet = _synth_plate(rng, state), _synth_wallet(rng)
        account, amount = _synth_account(rng), _synth_amount(rng)
        firm = _FRONT_BUSINESSES[int(rng.integers(len(_FRONT_BUSINESSES)))]
        border = _BORDER_POINTS[int(rng.integers(len(_BORDER_POINTS)))]
        handle = "@" + lead.split()[0].lower() + str(int(rng.integers(10, 99)))

        kind = n % 4
        if kind == 1:
            source_type, prefix = "FIR Narrative", "FIR"
            text = (
                f"{fir} registered at {city} City Police Station on {date_txt} under "
                f"case {case_id}. The complainant stated that accused {lead} approached "
                f"the victim through {firm} and offered employment in {city}. "
                f"Accused {lead} used mobile {phone} for contact and operated the social "
                f"media handle {handle}. "
                + (f"Co-accused {others[0]} arranged transport in vehicle {plate}. "
                   if others else "")
                + (f"Further, one {others[1]} was named as a facilitator. " if len(others) > 1 else "")
                + f"Handset bearing IMEI {imei} was recovered from the accused. "
                f"Payments totalling {amount} were traced to a/c {account} maintained at "
                f"{city}. Offence registered as {ctype}, graded severity {sev}. "
                f"Investigation is continuing."
            )
        elif kind == 2:
            source_type, prefix = "Surveillance Report", "SURV"
            text = (
                f"Surveillance note dated {date_txt}, {city} ({state}). Subject {lead} "
                f"was observed at {firm} between 21:40 and 23:15 hrs. "
                + (f"Subject met one {others[0]} and handed over a sealed packet. "
                   if others else "")
                + f"Vehicle {plate} was used to leave the premises towards {border}. "
                f"Number {phone} remained active on the same tower throughout. "
                + ("A country-made pistol was sighted in the subject's possession. "
                   if sev >= 4 and "Traffick" not in ctype else "")
                + (f"Movement is consistent with the {ctype} pattern recorded in case "
                   f"{case_id}. ")
                + f"Recommend continued watch and tower dump for {phone}."
            )
        elif kind == 3:
            source_type, prefix = "Social Media Intelligence", "SOCMINT"
            text = (
                f"Open-source monitoring report, {date_txt}. Handle {handle}, attributed "
                f"to {lead}, posted recruitment advertisements naming {firm} and offering "
                f"placement in {city}. "
                + (f"The account interacts frequently with a profile attributed to "
                   f"{others[0]}. " if others else "")
                + f"Contact number {phone} is listed in the profile bio. "
                f"Direct messages solicited advance payment of {amount} to "
                f"wallet {wallet}. Content is linked to case {case_id} ({ctype}). "
                f"Preservation request has been raised with the platform."
            )
        else:
            source_type, prefix = "Intelligence Agency Report", "INTREP"
            text = (
                f"Intelligence input dated {date_txt} concerning {ctype} activity in "
                f"{city}, {state}. {lead} is assessed as a principal node coordinating "
                f"movement through {border}. "
                + (f"Associates identified include {others[0]}"
                   + (f" and {others[1]}" if len(others) > 1 else "") + ". " if others else "")
                + f"Financial flows of approximately {amount} were routed via "
                f"a/c {account} and onward to wallet {wallet}. "
                f"Communications use {phone}, with handset IMEI {imei}. "
                + ("Consignment of heroin is suspected to move on the same route. "
                   if sev >= 4 and "Narcotic" in ctype else "")
                + f"Assessment confidence: moderate. Cross-reference case {case_id}."
            )

        rows.append({
            "doc_id": f"{prefix}-{year}-{n:04d}",
            "source_type": source_type,
            "title": f"{ctype} — {city}",
            "date": str(case["date"]),
            "city": city,
            "case_id": case_id,
            "text": text,
        })

    return pd.DataFrame(rows, columns=DOCUMENT_COLUMNS)


def write_sample_csvs(out_dir, n_persons: int = 220, n_syndicates: int = 6, seed: int = 42) -> dict:
    """Generate a dataset and persist it as CSVs. Returns the file paths."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = generate_sample_dataset(n_persons, n_syndicates, seed)
    paths = {}
    for name, df in data.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths
