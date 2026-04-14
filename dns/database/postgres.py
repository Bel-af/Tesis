import psycopg2
import os
from dotenv import load_dotenv

class Postgres:
    def __init__(self):
        # Cargar .env desde la raíz del proyecto (no depender del cwd)
        # override=False para permitir variables definidas por el servidor/contenedor
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=env_path, override=False)
        self.psql_user = os.getenv("PGSQL_USER", "tesis")
        self.psql_password = os.getenv("PGSQL_PASSWORD", "tesis")
        self.psql_db = os.getenv("PGSQL_DB", "tesis")
        self.psql_host = os.getenv("PGSQL_HOST", "localhost")
        self.psql_port = os.getenv("PGSQL_PORT", "5432")
        self.psql_connect_timeout = int(os.getenv("PGSQL_CONNECT_TIMEOUT", "8"))
        self.psql_sslmode = os.getenv("PGSQL_SSLMODE", "")
        print(f"Postgres config: user={self.psql_user}, db={self.psql_db}, host={self.psql_host}, port={self.psql_port}")
        self.connect()
        
    def connect(self):
        print("Connecting to PostgreSQL database")
        try:
            connection_kwargs = dict(
                user=self.psql_user,
                password=self.psql_password,
                host=self.psql_host,
                port=self.psql_port,
                database=self.psql_db,
                connect_timeout=self.psql_connect_timeout
            )
            if self.psql_sslmode:
                connection_kwargs["sslmode"] = self.psql_sslmode

            self.connection = psycopg2.connect(**connection_kwargs)
            print("Connection successful")
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            self.connection = None

    def getPacientes(self):
        """
        Obtiene lista de todos los pacientes.
        
        Returns:
            list: Lista de diccionarios con {id, nombre, edad, raza, diagnostico}
        """
        # Verificar conexion y reconectar si es necesario
        if self.connection is None or self.connection.closed:
            self.connect()
        
        if self.connection is not None:
            try:
                cursor = self.connection.cursor()
                # Log rápido para confirmar BD/usuario y conteo
                try:
                    cursor.execute("SELECT COUNT(*) FROM pacientes;")
                    total = cursor.fetchone()[0]
                    print(f"[DB {self.psql_db}] pacientes: {total}")
                except Exception as e:
                    print(f"Error contando pacientes: {e}")

                cursor.execute("SELECT id_paciente, nombre, edad, raza, \"Diagnostico\" FROM pacientes ORDER BY id_paciente;")
                pacientes = cursor.fetchall()
                cursor.close()
                # NO cerrar connection - se reutiliza
                return [{"id": row[0], "nombre": row[1], "edad": row[2], "raza": row[3], "diagnostico": row[4]} for row in pacientes]
            except Exception as e:
                print(f"Error obteniendo pacientes: {e}")
                return []
        else:
            return []

    def getPacienteById(self, id_paciente):
        """
        Obtiene un paciente por ID.

        Returns:
            dict o None
        """
        if self.connection is None or self.connection.closed:
            self.connect()

        if self.connection is not None:
            try:
                cursor = self.connection.cursor()
                cursor.execute(
                    "SELECT id_paciente, nombre, edad, raza, \"Diagnostico\" FROM pacientes WHERE id_paciente = %s;",
                    (id_paciente,)
                )
                row = cursor.fetchone()
                cursor.close()
                if row:
                    diagnostico = row[4] if len(row) > 4 else None
                    return {"id": row[0], "nombre": row[1], "edad": row[2], "raza": row[3], "diagnostico": diagnostico}
                return None
            except Exception as e:
                print(f"Error obteniendo paciente por ID: {e}")
                return None
        return None

    def updatePaciente(self, id_paciente, nombre=None, edad=None, raza=None, diagnostico=None):
        """
        Actualiza datos del paciente por ID.

        Returns:
            dict actualizado o None si no existe
        """
        if self.connection is None or self.connection.closed:
            self.connect()

        if self.connection is None:
            return None

        try:
            cursor = self.connection.cursor()

            # Obtener datos actuales
            cursor.execute(
                "SELECT nombre, edad, raza, \"Diagnostico\" FROM pacientes WHERE id_paciente = %s;",
                (id_paciente,)
            )
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return None

            current_nombre, current_edad, current_raza, current_diagnostico = row

            new_nombre = nombre if nombre is not None else current_nombre
            new_edad = edad if edad is not None else current_edad
            new_raza = raza if raza is not None else current_raza
            new_diagnostico = diagnostico if diagnostico is not None else current_diagnostico

            cursor.execute(
                "UPDATE pacientes SET nombre=%s, edad=%s, raza=%s, \"Diagnostico\"=%s WHERE id_paciente=%s;",
                (new_nombre, new_edad, new_raza, new_diagnostico, id_paciente)
            )
            self.connection.commit()
            cursor.close()

            return {
                "id": id_paciente,
                "nombre": new_nombre,
                "edad": new_edad,
                "raza": new_raza,
                "diagnostico": new_diagnostico
            }
        except Exception as e:
            print(f"Error actualizando paciente: {e}")
            self.connection.rollback()
            return None

    def deletePaciente(self, id_paciente):
        """
        Elimina un paciente por ID.

        Returns:
            True si eliminó, False si no existe
        """
        if self.connection is None or self.connection.closed:
            self.connect()

        if self.connection is None:
            return False

        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM pacientes WHERE id_paciente = %s;", (id_paciente,))
            deleted = cursor.rowcount > 0
            self.connection.commit()
            cursor.close()
            return deleted
        except Exception as e:
            print(f"Error eliminando paciente: {e}")
            self.connection.rollback()
            return False

    def crearPaciente(self, nombre, edad=None, raza=None, diagnostico=None):
        """
        Crea un paciente nuevo.

        Returns:
            dict con el paciente creado o None si falla.
        """
        if self.connection is None or self.connection.closed:
            self.connect()

        if self.connection is None:
            return None

        nombre_val = (nombre or '').strip()
        raza_val = (raza or '').strip()

        if not nombre_val or not raza_val:
            return None

        edad_val = None
        if edad is not None and str(edad).strip() != '':
            try:
                edad_val = int(edad)
            except (TypeError, ValueError):
                edad_val = None

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO pacientes (nombre, edad, raza, \"Diagnostico\") VALUES (%s, %s, %s, %s) RETURNING id_paciente",
                (nombre_val, edad_val, raza_val, diagnostico)
            )
            id_paciente = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            print(f"Paciente creado con ID: {id_paciente}")
            return {
                'id_paciente': id_paciente,
                'nombre': nombre_val,
                'edad': edad_val,
                'raza': raza_val,
                'diagnostico': diagnostico
            }
        except Exception as e:
            print(f"Error creando paciente: {e}")
            self.connection.rollback()
            return None
    
    # ============================================
    # NUEVOS METODOS PARA DNS (NO TOCAR ANTERIORES)
    # ============================================
    
    def guardarDiagnosticoDNS(self, id_paciente, nombre, edad, raza, resultado_dns, diagnostico=None):
        """
        Guarda resultado de diagnostico DNS en la base de datos.
        
        Args:
            id_paciente: ID del paciente (puede ser None para nuevo)
            nombre: Nombre del paciente
            edad: Edad del paciente
            raza: Raza del paciente
            resultado_dns: dict con resultado del analisis DNS
            
        Returns:
            dict con {id_examen, id_paciente, dns_probability_pct}
        """
        
        if self.connection is None or self.connection.closed:
            self.connect()
        
        if self.connection is None:
            print("Error: No hay conexion a la base de datos")
            return None

        nombre_val = (nombre or '').strip()
        raza_val = (raza or '').strip()

        if not nombre_val or not raza_val:
            return {
                'id_examen': None,
                'id_paciente': id_paciente,
                'dns_probability_pct': resultado_dns.get('dns_probability_pct', 0),
                'save_error': 'nombre y raza son obligatorios para guardar el paciente'
            }

        # Normalizar edad para evitar errores de tipo
        edad_val = None
        if edad is not None and str(edad).strip() != '':
            try:
                edad_val = int(edad)
            except (TypeError, ValueError):
                print(f"Advertencia: edad invalida '{edad}', se guardara como NULL")
                edad_val = None
        
        try:
            # Calcular diagnostico final antes de cualquier INSERT para evitar NULL en pacientes nuevos
            dns_probability_pct = float(resultado_dns.get('dns_probability_pct', 0) or 0)
            if diagnostico is None or str(diagnostico).strip() == '':
                diagnostico = 'DNS probable' if dns_probability_pct >= 50.0 else 'Sin indicadores de DNS'

            # Si no existe paciente, crearlo y confirmar en una transaccion separada
            if id_paciente is None:
                paciente_creado = self.crearPaciente(nombre_val, edad_val, raza_val, diagnostico=diagnostico)
                if paciente_creado is None:
                    return {
                        'id_examen': None,
                        'id_paciente': None,
                        'dns_probability_pct': resultado_dns.get('dns_probability_pct', 0),
                        'save_error': 'no se pudo crear el paciente'
                    }
                id_paciente = paciente_creado['id_paciente']

            num_samples = int(resultado_dns.get('num_samples', 0) or 0)
            lsb_uv = float(resultado_dns.get('metadata', {}).get('lsb_uv', 0) or 0)
            fs_hz = int(resultado_dns.get('fs_hz', 300) or 300)
            derivaciones = resultado_dns.get('derivations', []) or []

            # 1) Guardar SIEMPRE el estado del paciente y confirmar
            cursor = self.connection.cursor()
            cursor.execute(
                "UPDATE pacientes SET \"Diagnostico\" = %s WHERE id_paciente = %s",
                (diagnostico, id_paciente)
            )
            self.connection.commit()
            cursor.close()

            # 2) Intentar guardar metadatos + datos de examen (compatible con esquemas distintos)
            id_examen = None
            cursor = None
            try:
                cursor = self.connection.cursor()

                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='metadatos_examen'
                """)
                meta_columns = {row[0] for row in cursor.fetchall()}

                if {'generado_por_equipo', 'cantidad_muestras', 'tipo_dato', 'lsb_uv', 'frecuencia_muestreo_hz', 'derivaciones'}.issubset(meta_columns):
                    cursor.execute("""
                        INSERT INTO metadatos_examen
                        (generado_por_equipo, cantidad_muestras, tipo_dato, lsb_uv, frecuencia_muestreo_hz, derivaciones)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id_examen
                    """, (
                        'EasyG API',
                        num_samples,
                        'SIGNED INTEGER 32 BITS',
                        lsb_uv,
                        fs_hz,
                        ','.join(str(d) for d in derivaciones)
                    ))
                    id_examen = cursor.fetchone()[0]

                    cursor.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema='public' AND table_name='datos_examen'
                    """)
                    datos_columns = {row[0] for row in cursor.fetchall()}

                    if 'diagnostico_dns_pct' in datos_columns:
                        cursor.execute("""
                            INSERT INTO datos_examen (id_examen, id_paciente, diagnostico_dns_pct)
                            VALUES (%s, %s, %s)
                        """, (
                            id_examen,
                            id_paciente,
                            dns_probability_pct
                        ))
                    else:
                        cursor.execute("""
                            INSERT INTO datos_examen (id_examen, id_paciente, diagnostico)
                            VALUES (%s, %s, %s)
                        """, (
                            id_examen,
                            id_paciente,
                            dns_probability_pct
                        ))
                else:
                    # Esquema legado: primero insertar en datos_examen para obtener id_examen
                    cursor.execute("""
                        INSERT INTO datos_examen (id_paciente, diagnostico)
                        VALUES (%s, %s)
                        RETURNING id_examen
                    """, (
                        id_paciente,
                        dns_probability_pct
                    ))
                    id_examen = cursor.fetchone()[0]

                    cursor.execute("""
                        INSERT INTO metadatos_examen
                        (id_examen, generado_por, tipo_senal, dato_tipo, cantidad_muestras, "LSB", frecuencia_muestreo, orden)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        id_examen,
                        'EasyG API',
                        'ECG',
                        'SIGNED INTEGER 32 BITS',
                        num_samples,
                        lsb_uv,
                        fs_hz,
                        ','.join(str(d) for d in derivaciones)
                    ))

                self.connection.commit()
                cursor.close()

            except Exception as exam_error:
                print(f"Advertencia guardando examen DNS (se mantuvo estado del paciente): {exam_error}")
                self.connection.rollback()
                if cursor:
                    cursor.close()
                return {
                    'id_examen': None,
                    'id_paciente': id_paciente,
                    'dns_probability_pct': dns_probability_pct,
                    'save_error': str(exam_error)
                }

            print(f"Diagnostico DNS guardado: paciente={id_paciente}, examen={id_examen}, dns={dns_probability_pct:.1f}%")

            return {
                'id_examen': id_examen,
                'id_paciente': id_paciente,
                'dns_probability_pct': dns_probability_pct
            }

        except Exception as e:
            print(f"Error guardando diagnostico DNS: {e}")
            self.connection.rollback()
            return {
                'id_examen': None,
                'id_paciente': id_paciente,
                'dns_probability_pct': resultado_dns.get('dns_probability_pct', 0),
                'save_error': str(e)
            }
    
    def obtenerHistorialDNS(self, id_paciente):
        """
        Obtiene historial de diagnosticos DNS de un paciente.
        
        Args:
            id_paciente: ID del paciente
            
        Returns:
            list de diagnosticos ordenados por fecha
        """
        
        if self.connection is None:
            self.connect()
        
        if self.connection is None:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT 
                    de.id_examen,
                    de.diagnostico_dns_pct,
                    de.fecha_analisis,
                    me.cantidad_muestras,
                    me.frecuencia_muestreo_hz
                FROM datos_examen de
                JOIN metadatos_examen me ON de.id_examen = me.id_examen
                WHERE de.id_paciente = %s
                ORDER BY de.fecha_analisis DESC
            """, (id_paciente,))
            
            historial = cursor.fetchall()
            cursor.close()
            
            return [{
                'id_examen': row[0],
                'dns_probability_pct': row[1],
                'fecha_analisis': row[2].isoformat() if row[2] else None,
                'num_samples': row[3],
                'fs_hz': row[4]
            } for row in historial]
            
        except Exception as e:
            print(f"Error obteniendo historial DNS: {e}")
            return []
    
    def buscarPacientePorNombre(self, nombre, edad=None, raza=None):
        """
        Busca un paciente por nombre (y opcionalmente edad/raza).
        
        Args:
            nombre: Nombre del paciente
            edad: Edad (opcional, para desambiguar)
            raza: Raza (opcional, para desambiguar)
            
        Returns:
            dict con datos del paciente o None si no existe
        """
        
        if self.connection is None:
            self.connect()
        
        if self.connection is None:
            return None
        
        try:
            cursor = self.connection.cursor()

            nombre_val = (nombre or '').strip()
            raza_val = (raza or '').strip()
            
            # Busqueda flexible por nombre (case-insensitive)
            if edad is not None and raza is not None:
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s) AND edad = %s AND LOWER(raza) = LOWER(%s)
                    LIMIT 1
                """, (nombre_val, edad, raza_val))
            elif raza is not None and raza_val != '':
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s) AND LOWER(raza) = LOWER(%s)
                    LIMIT 1
                """, (nombre_val, raza_val))
            elif edad is not None:
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s) AND edad = %s
                    LIMIT 1
                """, (nombre_val, edad))
            else:
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s)
                    LIMIT 1
                """, (nombre_val,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'id_paciente': result[0],
                    'nombre': result[1],
                    'edad': result[2],
                    'raza': result[3]
                }
            return None
            
        except Exception as e:
            print(f"Error buscando paciente: {e}")
            return None
    
    def obtenerUltimoExamen(self, id_paciente):
        """
        Obtiene el ultimo examen DNS de un paciente con metricas completas.
        
        Args:
            id_paciente: ID del paciente
            
        Returns:
            dict con metricas del ultimo examen o None
        """
        
        if self.connection is None:
            self.connect()
        
        if self.connection is None:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT 
                    de.id_examen,
                    de.diagnostico_dns_pct,
                    de.fecha_analisis,
                    me.cantidad_muestras,
                    me.frecuencia_muestreo_hz
                FROM datos_examen de
                JOIN metadatos_examen me ON de.id_examen = me.id_examen
                WHERE de.id_paciente = %s
                ORDER BY de.fecha_analisis DESC
                LIMIT 1
            """, (id_paciente,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'id_examen': result[0],
                    'dns_probability_pct': result[1],
                    'fecha_analisis': result[2].isoformat() if result[2] else None,
                    'num_samples': result[3],
                    'fs_hz': result[4],
                    'duracion_s': result[3] / result[4] if result[4] > 0 else 0
                }
            return None
            
        except Exception as e:
            print(f"Error obteniendo ultimo examen: {e}")
            return None