from __future__ import annotations
from typing import Dict, Any
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_pdf_report(original_df, processed_df, decisions: Dict[str, Any], before_stats: Dict[str, Any], after_stats: Dict[str, Any]) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Rapport de Prétraitement (LLM + Heuristiques)", styles['Title']))
    elements.append(Spacer(1, 12))

    # Dataset overview
    elements.append(Paragraph("Vue d'ensemble du dataset (avant)", styles['Heading2']))
    dsb = before_stats['dataset']
    tbl_data = [["Metric", "Valeur"],
                ["Lignes", str(dsb['n_rows'])],
                ["Colonnes", str(dsb['n_cols'])],
                ["Taux de valeurs manquantes global", f"{dsb['missing_overall']:.3f}"],
                ["Répartition des types", str(dsb['type_counts'])]]
    tbl = Table(tbl_data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Vue d'ensemble du dataset (après)", styles['Heading2']))
    dsa = after_stats['dataset']
    tbl_data2 = [["Metric", "Valeur"],
                 ["Lignes", str(dsa['n_rows'])],
                 ["Colonnes", str(dsa['n_cols'])],
                 ["Taux de valeurs manquantes global", f"{dsa['missing_overall']:.3f}"],
                 ["Répartition des types", str(dsa['type_counts'])]]
    tbl2 = Table(tbl_data2, hAlign='LEFT')
    tbl2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
    ]))
    elements.append(tbl2)
    elements.append(Spacer(1, 12))

    # Drops
    elements.append(Paragraph("Colonnes supprimées", styles['Heading2']))
    drops = decisions.get('drops', [])
    elements.append(Paragraph(', '.join(drops) if drops else "Aucune", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Feature decisions
    elements.append(Paragraph("Décisions par feature", styles['Heading2']))
    feat_rows = [["Colonne", "Imputation", "Encodage", "Scaling", "Raison"]]
    for col, dec in decisions.get('features', {}).items():
        feat_rows.append([
            col,
            str(dec.get('imputation')),
            str(dec.get('encoding')),
            str(dec.get('scaling')),
            str(dec.get('reason', '')),
        ])
    feat_table = Table(feat_rows, hAlign='LEFT', repeatRows=1)
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(feat_table)

    doc.build(elements)
    buf.seek(0)
    return buf
