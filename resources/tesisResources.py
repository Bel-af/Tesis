from flask_restful import Resource
from flask import request
import json
import os
import tempfile
from database.postgres import Postgres
from database.ecg_processor import ECGProcessor

class Diagnosticar(Resource):
    def post(self):
        uploaded_file = request.files['file']
        nombre = request.form.get('nombre')
        edad = request.form.get('edad') 
        raza = request.form.get('raza')
        print(f"Archivo recibido: {uploaded_file.filename}, Nombre: {nombre}, Edad: {edad}, Raza: {raza}")
        
        # Guardar archivo temporal
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, uploaded_file.filename)
        uploaded_file.save(temp_filepath)
        print(f"Archivo guardado temporalmente en: {temp_filepath}")
        
        try:
            # Buscar si el paciente ya existe en la base de datos
            db = Postgres()
            paciente_existente = db.buscarPacientePorNombre(nombre, int(edad) if edad else None, raza)
            
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
                print(f"✓ Paciente nuevo (se creara en BD)")
            
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
            
            # Guardar en base de datos
            db_result = db.guardarDiagnosticoDNS(id_paciente, nombre, edad, raza, resultado_ecg)
            
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
                        interpretacion['recomendaciones'].append(
                            "Resultado calculado, pero no se pudo guardar examen en BD: {}".format(save_error)
                        )
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

        db = Postgres()
        actualizado = db.updatePaciente(id_paciente, nombre=nombre, edad=edad, raza=raza)
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
