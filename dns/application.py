from flask import Flask, jsonify, redirect, send_file
from flask_restful import Api, MethodNotAllowed, NotFound
from flask_cors import CORS
import os
from util.common import prefix, build_swagger_config_json, swagger_config_endpoint
from resources.swaggerConfig import SwaggerConfig
from resources.tesisResources import Diagnosticar, Pacientes, Paciente, HistorialClinico, DescargarPDF
from flask_swagger_ui import get_swaggerui_blueprint

# ============================================
# Main
# ============================================
application = Flask(__name__)
app = application
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configurar CORS más permisivo para downloads
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Disposition", "Content-Length"],
    "supports_credentials": False,
    "max_age": 3600
}})
api = Api(app, prefix=prefix, catch_all_404s=True)

# ============================================
# Swagger
# ============================================
build_swagger_config_json()
swaggerui_blueprint = get_swaggerui_blueprint(
    prefix,
    swagger_config_endpoint,
    config={
        'app_name': "Tesis API",
        "layout": "BaseLayout",
        "docExpansion": "none"
    },
)
app.register_blueprint(swaggerui_blueprint)

# ============================================
# Error Handler
# ============================================


@app.errorhandler(NotFound)
def handle_method_not_found(e):
    response = jsonify({"message": str(e)})
    response.status_code = 404
    return response


@app.errorhandler(MethodNotAllowed)
def handle_method_not_allowed_error(e):
    response = jsonify({"message": str(e)})
    response.status_code = 405
    return response


@app.route('/')
def redirect_to_prefix():
    if prefix != '':
        return redirect(prefix)


@app.route(prefix + '/') 
@app.route(prefix)
def serve_dashboard():
    """Sirve la interfaz HTML para análisis de ECG"""
    try:
        # Ruta del archivo HTML
        html_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
        if os.path.exists(html_path):
            return send_file(html_path)
        else:
            return jsonify({"error": "Interfaz HTML no encontrada"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route(prefix + '/descargar-pdf/<path:filename>')
def serve_pdf(filename):
    """Sirve archivos PDF del directorio de reportes
    
    Uso:
    - Desde navegador: 
      <a href="/dns/descargar-pdf/report_xxx_xxx.pdf" download>Descargar</a>
    - Desde JavaScript:
      fetch('/dns/descargar-pdf/report_xxx_xxx.pdf')
        .then(r => r.blob())
        .then(blob => ...)
    """
    try:
        try:
            from ecg_pipeline.config import REPORTS_DIR
        except (ImportError, AttributeError):
            REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
        
        # Validar que el filename no contiene caracteres peligrosos
        if '/' in filename or '\\' in filename or '..' in filename:
            return {"error": "Nombre de archivo inválido"}, 400
        
        # Construir ruta completa
        pdf_path = os.path.join(REPORTS_DIR, filename)
        
        # Verificar que existe el archivo
        if not os.path.exists(pdf_path):
            return {"error": f"Archivo no encontrado: {filename}"}, 404
        
        # Verificar que el archivo está dentro del directorio de reports (seguridad)
        if not os.path.abspath(pdf_path).startswith(os.path.abspath(REPORTS_DIR)):
            return {"error": "Acceso denegado"}, 403
        
        # Obtener tamaño del archivo para content-length
        file_size = os.path.getsize(pdf_path)
        
        # Crear response con headers correctos para download
        response = send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
        # Agregar headers adicionales para CORS y cache
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{filename}'
        response.headers['Cache-Control'] = 'no-cache, must-revalidate'
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Length'] = file_size
        
        return response
    except Exception as e:
        print(f"Error sirviendo PDF: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error al descargar: {str(e)}"}, 500


# ============================================
# Add Resource
# ============================================
# GET swagger config
api.add_resource(SwaggerConfig, '/swagger-config')
api.add_resource(Diagnosticar, '/diagnosticar')
api.add_resource(Pacientes, '/pacientes', '/pacientes/')
api.add_resource(Paciente, '/pacientes/<int:id_paciente>')
api.add_resource(DescargarPDF, '/descargar-pdf/<string:filename>')
#api.add_resource(HistorialClinico, '/historial/<int:id>')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
