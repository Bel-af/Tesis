from flask_restful import Resource
from flask import request, send_file
import json
import os
import tempfile
from datetime import datetime
from database.postgres import Postgres
from database.ecg_processor import ECGProcessor
from ecg_pipeline.reporter import generate_pdf_report

class Diagnosticar(Resource):
    def post(self):
        uploaded_file = request.files['file']
        nombre = (request.form.get('nombre') or '').strip()
        edad = request.form.get('edad')
        raza = (request.form.get('raza') or '').strip()

        if not nombre or not raza:
            return {
                "estado": "error",
                "recomendaciones": ["nombre y raza son obligatorios para registrar el paciente"]
            }, 400

        edad_val = None
        if edad is not None and str(edad).strip() != '':
            try:
                edad_val = int(edad)
            except (TypeError, ValueError):
                return {
                    "estado": "error",
                    "recomendaciones": ["edad invalida: debe ser un numero entero"]
                }, 400
        print(f"Archivo recibido: {uploaded_file.filename}, Nombre: {nombre}, Edad: {edad}, Raza: {raza}")
        
        # Guardar archivo temporal
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, uploaded_file.filename)
        uploaded_file.save(temp_filepath)
        print(f"Archivo guardado temporalmente en: {temp_filepath}")
        
        try:
            # Buscar si el paciente ya existe en la base de datos
            db = Postgres()
            paciente_existente = db.buscarPacientePorNombre(nombre, edad_val, raza)
            
            if paciente_existente:
                id_paciente = paciente_existente['id_paciente']
                print(f"✓ Paciente encontrado en BD: ID={id_paciente}, {paciente_existente['nombre']}")
                
                # Obtener ultimo examen para comparacion
                ultimo_examen = db.obtenerUltimoExamen(id_paciente)
                if ultimo_examen:
                    print(f"  - Ultimo examen: {ultimo_examen['fecha_analisis']}, DNS: {ultimo_examen['dns_probability_pct']:.1f}%")
            else:
                id_paciente = None
                ultimo_examen = None
                print("✓ Paciente nuevo detectado; se guardara junto con el diagnostico")
            
            # Procesar ECG con nuestro algoritmo DNS
            processor = ECGProcessor()
            resultado_ecg = processor.process_ecg_file(temp_filepath)
            
            # Obtener interpretacion clinica basica
            interpretacion = processor.get_clinical_interpretation(resultado_ecg)

            # Si el procesamiento fallo, retornar error real sin guardar en BD
            if 'error' in resultado_ecg or interpretacion.get('estado') == 'error':
                print(f"✗ Error en diagnostico ECG: {interpretacion.get('recomendaciones', [])}")
                return interpretacion
            
            # Si hay historico, agregar analisis comparativo
            if ultimo_examen and 'error' not in resultado_ecg:
                comparacion = processor.get_comparative_analysis(resultado_ecg, ultimo_examen)
                if comparacion:
                    interpretacion['comparacion'] = comparacion
                    
                    # Agregar alertas comparativas a recomendaciones
                    if comparacion.get('alertas_comparativas'):
                        interpretacion['recomendaciones'].extend(comparacion['alertas_comparativas'])
                    
                    print(f"  - Analisis comparativo: {comparacion['tendencia_descripcion']}")
                    print(f"  - Cambio DNS: {comparacion['cambio_dns_pct']:+.1f}%")

            report_base_name = os.path.splitext(os.path.basename(uploaded_file.filename))[0]
            report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"report_{report_base_name}_{report_timestamp}.pdf"
            pdf_download_url = f"/dns/descargar-pdf/{pdf_filename}"
            interpretacion['pdf_report_path'] = None
            interpretacion['pdf_download_url'] = pdf_download_url
            interpretacion['pdf_url'] = pdf_download_url

            try:
                pdf_report_path = generate_pdf_report(
                    uploaded_file.filename,
                    resultado_ecg.get('metadata', {}),
                    resultado_ecg,  # Pasar resultado_ecg completo como rr_analysis
                    resultado_ecg,  # Pasar resultado_ecg como features
                    resultado_ecg.get('dns_probability_pct', 0),
                    resultado_ecg.get('confidence_score_pct', 0),
                    interpretacion,
                    pdf_filename=pdf_filename,
                )
                if pdf_report_path:
                    # Actualizar la ruta real por si el generador normalizó el nombre
                    pdf_filename = os.path.basename(pdf_report_path)
                    pdf_download_url = f"/dns/descargar-pdf/{pdf_filename}"
                    interpretacion['pdf_report_path'] = pdf_report_path
                    interpretacion['pdf_download_url'] = pdf_download_url
                    interpretacion['pdf_url'] = pdf_download_url
            except Exception as pdf_error:
                print(f"✗ No se pudo generar el PDF: {pdf_error}")
                import traceback
                traceback.print_exc()
                interpretacion['pdf_report_path'] = None
            
            # Guardar en base de datos
            db_result = db.guardarDiagnosticoDNS(
                id_paciente,
                nombre,
                edad_val,
                raza,
                resultado_ecg,
                diagnostico=interpretacion.get('estado')
            )
            
            # Agregar IDs y estado de persistencia
            if db_result:
                if db_result.get('id_paciente') is not None:
                    interpretacion['id_paciente'] = db_result.get('id_paciente')
                if db_result.get('id_examen') is not None:
                    interpretacion['id_examen'] = db_result.get('id_examen')
                    interpretacion['db_save_ok'] = True
                else:
                    interpretacion['db_save_ok'] = False
                    save_error = db_result.get('save_error')
                    if save_error:
                        print(f"✗ Error guardando examen en BD: {save_error}")
            else:
                interpretacion['db_save_ok'] = False
                interpretacion['recomendaciones'].append(
                    "Resultado calculado, pero no se pudo confirmar guardado en BD"
                )
            
            dns_probability_pct = interpretacion.get('dns_probability_pct', 0)
            print(f"✓ Diagnostico DNS: {interpretacion.get('estado', 'N/A')} ({dns_probability_pct:.1f}%)")
            
            return interpretacion
            
        except Exception as e:
            print(f"Error procesando ECG: {e}")
            import traceback
            traceback.print_exc()
            return {
                "estado": "error",
                "recomendaciones": [f"Error al procesar ECG: {str(e)}"]
            }
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    

class Pacientes(Resource):
    def get(self):
        print("Obteniendo lista de pacientes")
        db = Postgres()
        listaPacientes=db.getPacientes()
        return listaPacientes

    def post(self):
        data = request.get_json(silent=True) or {}
        nombre = (data.get('nombre') or '').strip()
        edad = data.get('edad')
        raza = (data.get('raza') or '').strip()
        diagnostico = (data.get('diagnostico') or '').strip()

        if not nombre or not raza:
            return {"message": "nombre y raza son obligatorios"}, 400

        if not diagnostico:
            return {
                "message": "diagnostico es obligatorio. Para diagnostico automatico usa /diagnosticar"
            }, 400

        db = Postgres()
        paciente = db.crearPaciente(nombre, edad, raza, diagnostico=diagnostico)
        if paciente:
            return paciente, 201
        return {"message": "No se pudo crear el paciente"}, 500
    
class Paciente(Resource):
    def get(self,id_paciente:int):
        print(f"Obteniendo paciente con ID: {id_paciente}")
        db = Postgres()
        paciente = db.getPacienteById(id_paciente)
        if paciente:
            return paciente
        return {"message": "Paciente no encontrado"}, 404

    def put(self, id_paciente:int):
        print(f"Actualizando paciente con ID: {id_paciente}")
        data = request.get_json(silent=True) or {}
        nombre = data.get('nombre')
        edad = data.get('edad')
        raza = data.get('raza')
        diagnostico = data.get('diagnostico')

        db = Postgres()
        actualizado = db.updatePaciente(id_paciente, nombre=nombre, edad=edad, raza=raza, diagnostico=diagnostico)
        if actualizado:
            return actualizado
        return {"message": "Paciente no encontrado"}, 404

    def delete(self, id_paciente:int):
        print(f"Eliminando paciente con ID: {id_paciente}")
        db = Postgres()
        eliminado = db.deletePaciente(id_paciente)
        if eliminado:
            return {"message": "Paciente eliminado"}
        return {"message": "Paciente no encontrado"}, 404
    
class HistorialClinico(Resource):
    def get(self, id):
        historial={"id":id,"diagnosticos":[{"fecha":"2023-01-01","resultado":"saludable"}]}
        return historial


class DescargarPDF(Resource):
    """Endpoint para descargar archivos PDF generados"""
    def get(self, filename):
        """
        Descarga un archivo PDF por su nombre.
        Uso: GET /descargar-pdf/report_caso_demo_20260409_214845.pdf
        """
        try:
            # Obtener ruta de PDFs desde config
            from ecg_pipeline.config import REPORTS_DIR
        except (ImportError, AttributeError):
            REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
        
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
        
        try:
            # Servir el archivo como descarga
            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        except Exception as e:
            return {"error": f"Error al descargar: {str(e)}"}, 500
