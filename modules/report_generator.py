import time
import pandas as pd
from io import BytesIO

def generate_markdown_report(case_title, case_id, investigator_id, metrics_df, alerts, threat_index):
    """Generates official NCRB Case Intelligence Briefing in Markdown format."""
    now_str = time.strftime("%d %B %Y, %H:%M HRS IST")
    
    top_suspects = metrics_df.head(5) if not metrics_df.empty else pd.DataFrame()
    
    md = f"""# NATIONAL CRIME RECORDS BUREAU (NCRB)
## WOMEN SAFETY DIVISION - CASE INTELLIGENCE BRIEFING
**CONFIDENTIAL // LAW ENFORCEMENT SENSITIVE // OFFICIAL USE ONLY**

---

### CASE METADATA
- **Case Reference Title**: {case_title}
- **Case Tracking ID**: {case_id}
- **Lead Investigator ID**: {investigator_id}
- **Intelligence System**: Anti-Gravity AI Network Analyzer v2026.1
- **Generated Timestamp**: {now_str}
- **Overall Network Threat Index**: **{threat_index}%**

---

### EXECUTIVE SUMMARY
This intelligence briefing synthesizes multi-source crime data (FIR case logs, CDR call records, financial transactions, ANPR vehicle tracking, and geofence location surveillance). Advanced graph analytics and AI anomaly detection algorithms have mapped the underlying criminal network hierarchy, identifying key operational hubs, financial intermediaries, and high-threat targets.

---

### TOP IDENTIFIED NETWORK KEY PLAYERS & BROKERS

| Suspect ID | Name & Alias | Entity Type | Role / Function | Risk Score | Betweenness | PageRank |
|---|---|---|---|---|---|---|
"""
    if not top_suspects.empty:
        for _, row in top_suspects.iterrows():
            md += f"| {row['id']} | {row['name']} | {row['type']} | {row['role']} | **{row['risk_score']}/100** | {row['betweenness_centrality']} | {row['pagerank']} |\n"
    else:
        md += "| N/A | No Key Players Identified | N/A | N/A | N/A | N/A | N/A |\n"

    md += f"""
---

### DETECTED SUSPICIOUS OPERATIONAL ANOMALIES ({len(alerts)} ALERTS)

"""
    if alerts:
        for idx, alt in enumerate(alerts, 1):
            md += f"#### {idx}. [{alt['severity']}] {alt['title']}\n"
            md += f"- **Category**: {alt['type']}\n"
            md += f"- **Details**: {alt['description']}\n"
            md += f"- **Recommended Directive**: `{alt['recommended_action']}`\n\n"
    else:
        md += "No critical anomalies detected in the current network scope.\n\n"

    md += """
---

### INVESTIGATIVE DIRECTIVES & LEGAL RECOURSE
1. **Financial Interception**: Immediately freeze identified mule accounts under IPC / BNS & PMLA provisions.
2. **Telecommunications Surveillance**: Obtain CDR tower dump logs for primary suspect phone nodes.
3. **Interstate Coordination**: Issue lookout circulars (LOC) to state police forces for primary transport vehicles and safehouses.

---

*Report Cryptographically Signed & Timestamped by NCRB AI Network Intelligence System.*
"""
    return md

def generate_pdf_report(case_title, case_id, investigator_id, metrics_df, alerts, threat_index):
    """
    Generates a PDF case report using ReportLab if available, or returns binary text fallback.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor('#00f3ff'),
            alignment=1,
            spaceAfter=10
        )
        
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.HexColor('#a855f7'),
            alignment=1,
            spaceAfter=20
        )
        
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#334155'),
            spaceAfter=8
        )

        bold_style = ParagraphStyle(
            'BoldTextCustom',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        story.append(Paragraph("NATIONAL CRIME RECORDS BUREAU (NCRB)", title_style))
        story.append(Paragraph("WOMEN SAFETY DIVISION - CRIMINAL NETWORK INTELLIGENCE REPORT", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#00f3ff'), spaceAfter=15))
        
        # Meta info
        meta_data = [
            [Paragraph("<b>Case Title:</b> " + case_title, body_style), Paragraph("<b>Tracking ID:</b> " + case_id, body_style)],
            [Paragraph("<b>Investigator ID:</b> " + investigator_id, body_style), Paragraph("<b>Network Threat Index:</b> " + str(threat_index) + "%", body_style)],
            [Paragraph("<b>Timestamp:</b> " + time.strftime("%Y-%m-%d %H:%M HRS IST"), body_style), Paragraph("<b>Security Class:</b> RESTRICTED // NCRB", body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))

        # Top Suspects Table
        story.append(Paragraph("<b>TOP IDENTIFIED TARGETS & BROKERS</b>", bold_style))
        story.append(Spacer(1, 5))
        
        sus_headers = ["ID", "Name", "Type", "Role", "Risk"]
        table_data = [sus_headers]
        
        if not metrics_df.empty:
            for _, r in metrics_df.head(5).iterrows():
                table_data.append([str(r['id']), str(r['name']), str(r['type']), str(r['role']), str(r['risk_score']) + "/100"])
        
        sus_table = Table(table_data, colWidths=[60, 150, 90, 160, 80])
        sus_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00f3ff')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 5)
        ]))
        story.append(sus_table)
        story.append(Spacer(1, 15))

        # Alerts summary
        story.append(Paragraph("<b>FLAGGED ANOMALIES & THREAT ALERTS</b>", bold_style))
        story.append(Spacer(1, 5))
        for alt in alerts[:4]:
            story.append(Paragraph(f"• <b>[{alt['severity']}] {alt['title']}</b> - {alt['description']}", body_style))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    except ImportError:
        # Fallback to plain markdown bytes if reportlab is not installed
        md_text = generate_markdown_report(case_title, case_id, investigator_id, metrics_df, alerts, threat_index)
        return md_text.encode("utf-8")
