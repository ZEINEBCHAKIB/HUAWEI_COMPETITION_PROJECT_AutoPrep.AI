from __future__ import annotations
from typing import Dict, Any
import io

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch


def generate_pdf_report(original_df, processed_df, decisions: Dict[str, Any], before_stats: Dict[str, Any], after_stats: Dict[str, Any]) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Create custom styles for better text wrapping
    small_style = ParagraphStyle(
        'SmallText',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        wordWrap='CJK'
    )
    
    elements = []

    # Title
    elements.append(Paragraph("Data Preprocessing Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Dataset overview - Before
    elements.append(Paragraph("Dataset Overview (Before Preprocessing)", styles['Heading2']))
    dsb = before_stats['dataset']
    tbl_data = [["Metric", "Value"],
                ["Rows", str(dsb['n_rows'])],
                ["Columns", str(dsb['n_cols'])],
                ["Overall Missing Rate", f"{dsb['missing_overall']:.3f}"],
                ["Type Distribution", str(dsb['type_counts'])]]
    tbl = Table(tbl_data, colWidths=[2.5*inch, 2.5*inch], hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 12))

    # Dataset overview - After
    elements.append(Paragraph("Dataset Overview (After Preprocessing)", styles['Heading2']))
    dsa = after_stats['dataset']
    tbl_data2 = [["Metric", "Value"],
                 ["Rows", str(dsa['n_rows'])],
                 ["Columns", str(dsa['n_cols'])],
                 ["Overall Missing Rate", f"{dsa['missing_overall']:.3f}"],
                 ["Type Distribution", str(dsa['type_counts'])]]
    tbl2 = Table(tbl_data2, colWidths=[2.5*inch, 2.5*inch], hAlign='LEFT')
    tbl2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(tbl2)
    elements.append(Spacer(1, 12))

    # Global configuration
    elements.append(Paragraph("Global Configuration", styles['Heading2']))
    global_config = decisions.get('global', {})
    config_data = [
        ["Parameter", "Value"],
        ["Low Cardinality Threshold", str(global_config.get('low_card_threshold', 'N/A'))],
        ["High Missing Threshold", f"{global_config.get('high_missing_threshold', 'N/A')}"],
        ["Outlier Treatment", str(global_config.get('outliers', 'N/A'))],
        ["Scaling Enabled", str(global_config.get('scaling', 'N/A'))],
        ["Scaling Method", str(global_config.get('scaling_method', 'Auto-detect'))],
    ]
    config_table = Table(config_data, colWidths=[2.5*inch, 2.5*inch], hAlign='LEFT')
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(config_table)
    elements.append(Spacer(1, 12))

    # Dropped columns
    elements.append(Paragraph("Dropped Columns", styles['Heading2']))
    drops = decisions.get('drops', [])
    elements.append(Paragraph(', '.join(drops) if drops else "None", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Feature decisions - use landscape for better visibility
    elements.append(PageBreak())
    elements.append(Paragraph("Feature Processing Decisions", styles['Heading2']))
    elements.append(Spacer(1, 6))
    
    feat_rows = [["Column", "Imputation", "Encoding", "Scaling", "Reason"]]
    for col, dec in decisions.get('features', {}).items():
        feat_rows.append([
            Paragraph(str(col), small_style),
            Paragraph(str(dec.get('imputation', 'N/A')), small_style),
            Paragraph(str(dec.get('encoding', 'N/A')), small_style),
            Paragraph(str(dec.get('scaling', 'N/A')), small_style),
            Paragraph(str(dec.get('reason', 'N/A')), small_style),
        ])
    
    feat_table = Table(feat_rows, colWidths=[1.2*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1.6*inch], hAlign='LEFT', repeatRows=1)
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightblue]),
    ]))
    elements.append(feat_table)

    doc.build(elements)
    buf.seek(0)
    return buf
