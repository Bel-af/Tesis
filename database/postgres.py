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
            list: Lista de diccionarios con {id, nombre, edad, raza}
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

                cursor.execute("SELECT id_paciente, nombre, edad, raza FROM pacientes ORDER BY id_paciente;")
                pacientes = cursor.fetchall()
                cursor.close()
                # NO cerrar connection - se reutiliza
                return [{"id": row[0], "nombre": row[1], "edad": row[2], "raza": row[3]} for row in pacientes]
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
                    "SELECT id_paciente, nombre, edad, raza FROM pacientes WHERE id_paciente = %s;",
                    (id_paciente,)
                )
                row = cursor.fetchone()
                cursor.close()
                if row:
                    return {"id": row[0], "nombre": row[1], "edad": row[2], "raza": row[3]}
                return None
            except Exception as e:
                print(f"Error obteniendo paciente por ID: {e}")
                return None
        return None

    def updatePaciente(self, id_paciente, nombre=None, edad=None, raza=None):
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
                "SELECT nombre, edad, raza FROM pacientes WHERE id_paciente = %s;",
                (id_paciente,)
            )
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return None

            current_nombre, current_edad, current_raza = row

            new_nombre = nombre if nombre is not None else current_nombre
            new_edad = edad if edad is not None else current_edad
            new_raza = raza if raza is not None else current_raza

            cursor.execute(
                "UPDATE pacientes SET nombre=%s, edad=%s, raza=%s WHERE id_paciente=%s;",
                (new_nombre, new_edad, new_raza, id_paciente)
            )
            self.connection.commit()
            cursor.close()

            return {
                "id": id_paciente,
                "nombre": new_nombre,
                "edad": new_edad,
                "raza": new_raza
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
    
    # ============================================
    # NUEVOS METODOS PARA DNS (NO TOCAR ANTERIORES)
    # ============================================
    
    def guardarDiagnosticoDNS(self, id_paciente, nombre, edad, raza, resultado_dns):
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

        # Normalizar edad para evitar errores de tipo
        edad_val = None
        if edad is not None and str(edad).strip() != '':
            try:
                edad_val = int(edad)
            except (TypeError, ValueError):
                print(f"Advertencia: edad invalida '{edad}', se guardara como NULL")
                edad_val = None
        
        try:
            # Si no existe paciente, crearlo y confirmar en una transaccion separada
            if id_paciente is None:
                cursor = self.connection.cursor()
                cursor.execute(
                    "INSERT INTO pacientes (nombre, edad, raza) VALUES (%s, %s, %s) RETURNING id_paciente",
                    (nombre, edad_val, raza)
                )
                id_paciente = cursor.fetchone()[0]
                cursor.close()
                self.connection.commit()
                print(f"Paciente creado con ID: {id_paciente}")

            cursor = self.connection.cursor()

            # Crear registro en metadatos_examen
            cursor.execute("""
                INSERT INTO metadatos_examen 
                (generado_por_equipo, cantidad_muestras, tipo_dato, lsb_uv, frecuencia_muestreo_hz, derivaciones)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id_examen
            """, (
                'EasyG API',
                resultado_dns.get('num_samples', 0),
                'SIGNED INTEGER 32 BITS',
                resultado_dns.get('metadata', {}).get('lsb_uv', 0),
                resultado_dns.get('fs_hz', 300),
                ','.join(resultado_dns.get('derivations', []))
            ))
            
            id_examen = cursor.fetchone()[0]
            print(f"Examen creado con ID: {id_examen}")
            
            # Crear registro en datos_examen con probabilidad DNS
            cursor.execute("""
                INSERT INTO datos_examen (id_examen, id_paciente, diagnostico_dns_pct)
                VALUES (%s, %s, %s)
            """, (
                id_examen,
                id_paciente,
                resultado_dns.get('dns_probability_pct', 0)
            ))
            
            self.connection.commit()
            cursor.close()
            
            print(f"Diagnostico DNS guardado: paciente={id_paciente}, examen={id_examen}, dns={resultado_dns.get('dns_probability_pct', 0):.1f}%")
            
            return {
                'id_examen': id_examen,
                'id_paciente': id_paciente,
                'dns_probability_pct': resultado_dns.get('dns_probability_pct', 0)
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
            
            # Busqueda flexible por nombre (case-insensitive)
            if edad is not None and raza is not None:
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s) AND edad = %s AND LOWER(raza) = LOWER(%s)
                    LIMIT 1
                """, (nombre, edad, raza))
            elif edad is not None:
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s) AND edad = %s
                    LIMIT 1
                """, (nombre, edad))
            else:
                cursor.execute("""
                    SELECT id_paciente, nombre, edad, raza
                    FROM pacientes
                    WHERE LOWER(nombre) = LOWER(%s)
                    LIMIT 1
                """, (nombre,))
            
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