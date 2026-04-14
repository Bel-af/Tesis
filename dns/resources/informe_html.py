#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
informe_html.py - Helpers para pagina HTML del informe y descarga de PDF.

Este archivo NO modifica la app existente. Se puede importar despues para
registrar rutas o reutilizar la plantilla HTML del informe.
"""

from __future__ import annotations

import glob
import os
from urllib.parse import quote

from flask import request, send_file


DEFAULT_PREFIX = "/dns"


def _safe_basename(name: str) -> str:
    return os.path.basename((name or "").strip()).replace("..", "")


def _resolve_reports_dir() -> str:
    try:
        from ecg_pipeline.config import REPORTS_DIR
        return REPORTS_DIR
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "reports")


def build_informe_html(filename: str, pdf_exists: bool, download_url: str) -> str:
    estado_pdf = f"PDF encontrado: {filename}" if pdf_exists else "Aún no aparece el PDF generado o no se indicó un archivo válido."
    disabled_attr = "" if pdf_exists else 'aria-disabled="true"'
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
    <p class="muted">{estado_pdf}</p>
    <a class="btn" href="{download_url}" {disabled_attr}>Descargar PDF</a>
    <p class="muted">Si no se actualiza, vuelve a intentar o recarga la página.</p>
  </div>
</body>
</html>"""


def build_download_url(filename: str, prefix: str = DEFAULT_PREFIX) -> str:
    safe_filename = _safe_basename(filename)
    return f"{prefix}/descargar-pdf/{quote(safe_filename)}"


def build_landing_url(filename: str, prefix: str = DEFAULT_PREFIX) -> str:
    safe_filename = _safe_basename(filename)
    return f"{prefix}/informe?file={quote(safe_filename)}"


def serve_informe():
    """Flask view: pagina HTML intermedia del informe."""
    reports_dir = _resolve_reports_dir()
    filename = _safe_basename(request.args.get("file", ""))
    pdf_path = os.path.join(reports_dir, filename) if filename else ""
    pdf_exists = bool(filename) and os.path.exists(pdf_path)
    download_url = build_download_url(filename) if pdf_exists else "#"
    return build_informe_html(filename, pdf_exists, download_url)


def serve_pdf(filename: str):
    """Flask view: descarga directa del PDF."""
    reports_dir = _resolve_reports_dir()
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


def find_latest_report(slug: str) -> str | None:
    """Busca el último PDF generado que coincida con un slug de archivo."""
    reports_dir = _resolve_reports_dir()
    safe_slug = _safe_basename(slug)
    pattern = os.path.join(reports_dir, f"report_{safe_slug}_*.pdf")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def register_informe_routes(app, prefix: str = DEFAULT_PREFIX):
    """Registra las rutas /informe y /descargar-pdf en una app Flask."""

    @app.route(f"{prefix}/informe")
    def _informe():
        return serve_informe()

    @app.route(f"{prefix}/descargar-pdf/<path:filename>")
    def _download_pdf(filename):
        return serve_pdf(filename)

    @app.route(f"{prefix}/informe/<path:slug>")
    def _informe_slug(slug):
        latest = find_latest_report(slug)
        if latest:
            pdf_name = os.path.basename(latest)
            return build_informe_html(pdf_name, True, build_download_url(pdf_name, prefix=prefix))
        return build_informe_html("", False, "#")

    return app
