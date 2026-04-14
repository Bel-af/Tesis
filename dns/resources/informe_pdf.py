#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
informe_pdf.py

Modulo aislado para:
- servir una pagina HTML de informe
- servir la descarga directa del PDF
- construir URLs publicas para el frontend

Este archivo no modifica la app actual por si sola.
Para activarlo, se debe importar y registrar su blueprint o funciones
en application.py.
"""

from __future__ import annotations

import os
from urllib.parse import quote

from flask import Blueprint, request, send_file


def _safe_basename(value: str) -> str:
    return os.path.basename((value or "").strip()).replace("..", "")


def _reports_dir() -> str:
    try:
        from ecg_pipeline.config import REPORTS_DIR
        return REPORTS_DIR
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "reports")


def build_download_url(filename: str, prefix: str = "/dns") -> str:
    safe_filename = _safe_basename(filename)
    return f"{prefix}/descargar-pdf/{quote(safe_filename)}"


def build_html_page(filename: str, pdf_exists: bool, download_url: str) -> str:
    status = (
        f"PDF encontrado: {filename}"
        if pdf_exists
        else "Aún no aparece el PDF generado o no se indicó un archivo válido."
    )
    disabled = "" if pdf_exists else 'aria-disabled="true"'

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Informe ECG</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f7fb; color: #1f2937; }}
    .wrap {{ max-width: 760px; margin: 8vh auto; background: white; padding: 32px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.08); }}
    h1 {{ margin-top: 0; font-size: 1.6rem; }}
    p {{ line-height: 1.5; }}
    a.btn {{ display: inline-block; margin-top: 16px; padding: 12px 18px; border-radius: 10px; background: #0ea5e9; color: white; text-decoration: none; font-weight: 700; }}
    .muted {{ color: #6b7280; font-size: .95rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Informe del análisis</h1>
    <p><strong>Archivo:</strong> {filename or 'No especificado'}</p>
    <p class="muted">{status}</p>
    <a class="btn" href="{download_url}" {disabled}>Descargar PDF</a>
    <p class="muted">Si no se actualiza, vuelve a intentar o recarga la página.</p>
  </div>
</body>
</html>"""


def serve_pdf(filename: str):
    reports_dir = _reports_dir()
    safe_filename = _safe_basename(filename)

    if not safe_filename:
        return {"error": "Nombre de archivo inválido"}, 400

    pdf_path = os.path.join(reports_dir, safe_filename)
    if not os.path.exists(pdf_path):
        return {"error": f"Archivo no encontrado: {safe_filename}"}, 404

    if not os.path.abspath(pdf_path).startswith(os.path.abspath(reports_dir)):
        return {"error": "Acceso denegado"}, 403

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=safe_filename,
    )


def serve_informe():
    filename = _safe_basename(request.args.get("file", ""))
    reports_dir = _reports_dir()
    pdf_path = os.path.join(reports_dir, filename) if filename else ""
    pdf_exists = bool(filename) and os.path.exists(pdf_path)
    download_url = build_download_url(filename) if pdf_exists else "#"
    return build_html_page(filename, pdf_exists, download_url)


def create_informe_blueprint(prefix: str = "/dns") -> Blueprint:
    """Crea un blueprint aislado con las rutas del informe."""
    blueprint = Blueprint("informe_pdf", __name__)

    @blueprint.route(f"{prefix}/informe")
    def informe_page():
        return serve_informe()

    @blueprint.route(f"{prefix}/descargar-pdf/<path:filename>")
    def download_pdf(filename: str):
        return serve_pdf(filename)

    return blueprint


def build_frontend_pdf_url(filename: str, prefix: str = "/dns") -> str:
    """URL lista para devolver al frontend (modo relativo)."""
    return f"{prefix}/informe?file={quote(_safe_basename(filename))}"
