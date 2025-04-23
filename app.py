#!/usr/bin/env python3
"""
TheModernPromethease - Aplicación Web de Análisis Genético

Esta aplicación web permite a los usuarios subir sus datos genéticos y ejecutar
el pipeline completo de análisis genético de manera sencilla e interactiva.

La aplicación:
1. Permite subir archivos de datos genéticos (incluyendo conversión de MyHeritage)
2. Ejecuta automáticamente el pipeline de análisis
3. Muestra los resultados de manera clara y amigable
"""

import streamlit as st
import os
import subprocess
import time
from pathlib import Path
import shutil
import pandas as pd
import tempfile
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import uuid
import logging
from src.genomic_database_manager import GenomicDatabaseManager
from src.database_optimizer import DatabaseOptimizer
from typing import Tuple
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuración Inicial
# =============================================================================

# Configurar el título y la descripción de la página
st.set_page_config(
    page_title="TheModernPromethease",
    page_icon="🧬",
    layout="centered"
)

# Título principal
st.title("🧬 TheModernPromethease")
st.markdown("""
    ### Análisis Genético Personalizado
    
    Sube tus datos genéticos para obtener un análisis completo de tu predisposición genética
    a diferentes condiciones de salud.
""")

# =============================================================================
# Funciones Auxiliares
# =============================================================================

def generar_id_unico():
    """Genera un ID único basado en timestamp y UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"

def verificar_estructura_directorios():
    """Verifica y crea la estructura de directorios necesaria."""
    directorios = [
        'data/raw', 
        'data/processed', 
        'data/myheritage',
        'reports'
    ]
    for directorio in directorios:
        Path(directorio).mkdir(parents=True, exist_ok=True)
        logger.info(f"Verified directory exists: {directorio}")

def validate_myheritage_data(df: pd.DataFrame) -> bool:
    """Validate MyHeritage data format and content."""
    try:
        # Check required columns
        required_columns = ['RSID', 'CHROMOSOME', 'POSITION', 'RESULT']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ Error: Columnas faltantes. Se encontraron: {df.columns.tolist()}")
            st.error(f"❌ Se esperaban las columnas: {', '.join(required_columns)}")
            return False
        
        # Validate chromosome values
        valid_chromosomes = set(str(i) for i in range(1, 23)) | {'X', 'Y', 'MT'}
        invalid_chromosomes = set(df['CHROMOSOME'].astype(str)) - valid_chromosomes
        if invalid_chromosomes:
            st.error(f"❌ Error: Valores inválidos en CHROMOSOME: {invalid_chromosomes}")
            st.error("Los valores válidos son números del 1 al 22, X, Y o MT")
            return False
        
        # Validate position values
        if not pd.api.types.is_numeric_dtype(df['POSITION']):
            st.error("❌ Error: La columna POSITION debe contener valores numéricos")
            return False
        
        # Validate RSID format
        if not df['RSID'].str.startswith('rs').all():
            st.error("❌ Error: Todos los valores de RSID deben comenzar con 'rs'")
            return False
        
        # Validate genotype format
        valid_genotypes = {'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 
                          'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT', 
                          '--', 'II', 'DD', 'DI', 'ID'}
        invalid_genotypes = set(df['RESULT'].astype(str)) - valid_genotypes
        if invalid_genotypes:
            st.error(f"❌ Error: Valores de genotipo inválidos encontrados: {invalid_genotypes}")
            st.error("Los valores válidos son combinaciones de A, C, G, T o los valores especiales: --, II, DD, DI, ID")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ Error durante la validación: {str(e)}")
        return False

def convert_myheritage(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Convert MyHeritage data to required format."""
    try:
        # Generate unique session ID
        session_id = generar_id_unico()
        logger.info(f"Generated session ID: {session_id}")
        
        # Ensure directories exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/myheritage", exist_ok=True)
        logger.info("Verified directory structure")
        
        # Convert chromosome values to string and standardize format
        df['CHROMOSOME'] = df['CHROMOSOME'].astype(str).str.upper()
        
        # Ensure position is numeric
        df['POSITION'] = pd.to_numeric(df['POSITION'], errors='coerce')
        
        # Create the output DataFrame with required columns
        converted_df = pd.DataFrame({
            'rsid': df['RSID'],
            'chromosome': df['CHROMOSOME'],
            'position': df['POSITION'],
            'genotype': df['RESULT']
        })
        
        # Remove any rows with missing values
        converted_df = converted_df.dropna()
        logger.info(f"Converted data: {len(converted_df)} rows")
        
        # Save the converted data in Parquet format
        output_path = f"data/raw/usuario_{session_id}.parquet"
        logger.info(f"Guardando archivo Parquet en: {output_path}")
        converted_df.to_parquet(output_path, index=False)
        
        # Verify Parquet file was created
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"No se pudo crear el archivo Parquet en {output_path}")
        logger.info(f"Parquet file created successfully: {output_path}")
        
        # Save a copy in JSON format for MyHeritage specific processing
        myheritage_data = {
            'variants': converted_df.to_dict('records'),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        myheritage_path = os.path.join("data", "myheritage", f"myheritage_{session_id}.json")
        logger.info(f"Guardando archivo JSON en: {myheritage_path}")
        
        try:
            with open(myheritage_path, 'w') as f:
                json.dump(myheritage_data, f, indent=2)
            logger.info(f"JSON data written to file: {myheritage_path}")
        except Exception as e:
            logger.error(f"Error writing JSON file: {str(e)}")
            raise
            
        # Verify JSON file was created
        if not os.path.exists(myheritage_path):
            raise FileNotFoundError(f"No se pudo crear el archivo JSON en {myheritage_path}")
        logger.info(f"JSON file created successfully: {myheritage_path}")
            
        st.success(f"✅ Datos convertidos y guardados exitosamente")
        st.info(f"📁 Archivo Parquet: {output_path}")
        st.info(f"📁 Archivo JSON: {myheritage_path}")
        st.info(f"ID de sesión: {session_id}")
        
        return converted_df, session_id
        
    except Exception as e:
        logger.error(f"❌ Error al convertir los datos: {str(e)}")
        st.error(f"❌ Error al convertir los datos: {str(e)}")
        raise

def validar_archivo(archivo):
    """Valida el formato del archivo de datos genéticos."""
    try:
        # Leer las primeras líneas del archivo
        contenido = archivo.getvalue().decode('utf-8').split('\n')
        if len(contenido) < 2:
            return False, "El archivo está vacío o no tiene el formato correcto"
        
        # Verificar encabezados
        encabezados = contenido[0].strip().split('\t')
        columnas_requeridas = ['rsid', 'chromosome', 'position', 'genotype']
        
        if not all(col in encabezados for col in columnas_requeridas):
            return False, "El archivo no contiene todas las columnas requeridas (rsid, chromosome, position, genotype)"
        
        return True, "Archivo válido"
    except Exception as e:
        return False, f"Error al validar el archivo: {str(e)}"

def ejecutar_pipeline(df_convertido, gwas_data, clinvar_data, dbsnp_data):
    """Ejecuta el pipeline de análisis genético."""
    try:
        # Configurar indicadores de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Verificar estructura básica
        status_text.text("🔄 Verificando estructura del pipeline...")
        if not os.access('run_pipeline.sh', os.X_OK):
            st.warning("⚠️ El script run_pipeline.sh no es ejecutable. Intentando hacerlo ejecutable...")
            os.chmod('run_pipeline.sh', 0o755)
        
        # Verificar archivos y permisos
        status_text.text("🔄 Verificando archivos y permisos...")
        archivo_datos = 'data/raw/usuario_adaptado.txt'
        if not os.path.exists(archivo_datos):
            st.error(f"❌ Error: No se encontró el archivo de datos en {archivo_datos}")
            return False
        
        # Crear directorios necesarios
        directorios = ['data/processed', 'reports']
        for directorio in directorios:
            if not os.path.exists(directorio):
                os.makedirs(directorio)
            if not os.access(directorio, os.W_OK):
                st.error(f"❌ Error: No hay permisos de escritura en el directorio {directorio}")
                return False
        
        # Ejecutar pipeline con logging detallado
        status_text.text("🔄 Iniciando pipeline de análisis (esto puede tomar varios minutos)...")
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as log_file:
            # Configurar variables de entorno para el pipeline
            env = {
                **os.environ,
                'LOG_FILE': log_file.name,
                'SHOW_PROGRESS': 'true',  # Indicar al pipeline que muestre progreso
                'GWAS_DATA': gwas_data,
                'CLINVAR_DATA': clinvar_data,
                'DBSNP_DATA': dbsnp_data
            }
            
            # Ejecutar pipeline en segundo plano para no bloquear la interfaz
            proceso = subprocess.Popen(
                ['./run_pipeline.sh'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Monitorear progreso
            while True:
                # Leer log para actualizar progreso
                log_file.seek(0)
                log_content = log_file.read()
                
                # Actualizar interfaz
                if "Iniciando extracción de datos" in log_content:
                    progress_bar.progress(0.2)
                elif "Calculando PRS" in log_content:
                    progress_bar.progress(0.4)
                elif "Generando reporte" in log_content:
                    progress_bar.progress(0.6)
                elif "Generando interpretación con IA" in log_content:
                    progress_bar.progress(0.8)
                
                # Verificar si el proceso terminó
                if proceso.poll() is not None:
                    break
                
                # Esperar un momento antes de la siguiente actualización
                time.sleep(1)
            
            # Obtener resultados finales
            stdout, stderr = proceso.communicate()
            
            if proceso.returncode != 0:
                st.error("❌ Error en el pipeline:")
                st.error(f"Código de salida: {proceso.returncode}")
                st.error("Salida estándar:")
                st.text(stdout)
                st.error("Error estándar:")
                st.text(stderr)
                st.error("Log del pipeline:")
                st.text(log_content)
                return False
            
            # Mostrar log de éxito
            progress_bar.progress(1.0)
            status_text.text("✅ Pipeline ejecutado exitosamente")
            st.info("Log del pipeline:")
            st.text(log_content)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        st.error("Por favor, verifica los logs para más detalles")
        return False

def leer_resultados():
    """Lee y retorna los resultados del análisis."""
    try:
        with open('reports/prs_result.txt', 'r') as f:
            prs_resultado = f.read()
        with open('reports/prs_interpretation.txt', 'r') as f:
            interpretacion = f.read()
        return prs_resultado, interpretacion
    except Exception as e:
        st.error(f"Error al leer los resultados: {str(e)}")
        return None, None

def procesar_datos_geneticos(df, session_id):
    """Procesa los datos genéticos y genera el informe.
    
    El proceso incluye:
    1. Extracción y validación de SNPs
    2. Consulta a bases de datos genéticas
    3. Análisis clínico de variantes
    4. Generación del reporte final
    
    Args:
        df (pd.DataFrame): DataFrame con los datos genéticos
        session_id (str): ID único de la sesión
        
    Returns:
        bool: True si el proceso fue exitoso, False en caso contrario
    """
    try:
        # Mostrar indicador de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Extraer y validar SNPs
        status_text.text("🔄 Extrayendo y validando SNPs...")
        progress_bar.progress(0.2)
        
        # Validar formato de SNPs
        valid_snps = df[
            (df['rsid'].str.startswith('rs')) &
            (df['chromosome'].isin([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT'])) &
            (df['position'].notna())
        ]
        
        if len(valid_snps) == 0:
            raise ValueError("No se encontraron SNPs válidos en los datos")
        
        st.info(f"✅ Se encontraron {len(valid_snps)} SNPs válidos")
        
        # 2. Consultar bases de datos genéticas
        status_text.text("🔄 Consultando bases de datos genéticas...")
        progress_bar.progress(0.4)
        
        # Inicializar gestor de base de datos
        db_manager = GenomicDatabaseManager()
        
        # Obtener lista de rs_ids
        rs_ids = valid_snps['rsid'].tolist()
        
        # Procesar datos usando el método que sabemos que funciona
        report_df = db_manager.procesar_datos_geneticos(rs_ids)
        
        if report_df is None or report_df.empty:
            raise ValueError("No se pudieron obtener datos de las bases de datos genéticas")
        
        # 3. Generar informe detallado
        status_text.text("🔄 Generando informe detallado...")
        progress_bar.progress(0.6)
        
        # Crear directorio de reports si no existe
        os.makedirs('reports', exist_ok=True)
        
        # Guardar el reporte en formato parquet
        report_path = f"reports/report_{session_id}.parquet"
        report_df.to_parquet(report_path)
        
        # Verificar que el archivo se creó correctamente
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"No se pudo crear el archivo de reporte en {report_path}")
        
        st.info(f"Reporte guardado en: {report_path}")
        
        # 4. Generar informe narrativo
        status_text.text("🔄 Generando informe narrativo...")
        progress_bar.progress(0.8)
        
        # Analizar hallazgos significativos
        significant_findings = report_df[
            (report_df['clinvar_clinical_significance'] != 'Unknown') |
            (report_df['gwas_p_value'] < 0.05)
        ]
        
        st.info(f"✅ Se encontraron {len(significant_findings)} hallazgos significativos")
        
        # 5. Guardar resultados
        status_text.text("🔄 Guardando resultados...")
        progress_bar.progress(1.0)
        
        # Mostrar enlace al informe
        st.success("✅ Procesamiento completado")
        st.markdown(f"""
            ### 📊 Informe Generado
            
            Tu informe detallado está listo. Puedes verlo haciendo clic en el siguiente enlace:
            
            [Ver Informe Detallado](/report?id={session_id})
            
            El informe contiene:
            - {len(valid_snps)} SNPs analizados
            - {len(report_df)} variantes encontradas
            - {len(significant_findings)} hallazgos significativos
            - Tabla interactiva con todos los resultados
            - Opción para descargar el informe completo
        """)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {str(e)}")
        logger.error(f"Error al procesar datos genéticos: {str(e)}")
        return False

# =============================================================================
# Interfaz Principal
# =============================================================================

def main():
    """Función principal de la aplicación."""
    # Verificar estructura de directorios
    verificar_estructura_directorios()
    
    # Sección de subida de archivos
    st.header("📤 Sube tus datos genéticos")
    st.markdown("""
        Por favor, sube tu archivo de datos genéticos de MyHeritage.
        El archivo debe estar en formato CSV con las columnas: RSID, CHROMOSOME, POSITION, RESULT
        
        ### Instrucciones:
        1. Descarga tu archivo de datos genéticos de MyHeritage
        2. Asegúrate de que el archivo tenga las columnas correctas
        3. Sube el archivo usando el botón de abajo
    """)
    
    archivo_subido = st.file_uploader(
        "Selecciona tu archivo MyHeritage",
        type=['csv', 'txt', 'tsv'],
        help="Archivo de datos genéticos de MyHeritage (CSV o TSV)"
    )
    
    if archivo_subido is not None:
        try:
            # Mostrar información del archivo
            st.info(f"📄 Archivo subido: {archivo_subido.name}")
            
            # Leer el archivo CSV, ignorando las líneas de comentario
            df = pd.read_csv(
                archivo_subido,
                comment='#',  # Ignorar líneas que comienzan con #
                skip_blank_lines=True,
                dtype={
                    'RSID': str,
                    'CHROMOSOME': str,
                    'POSITION': int,
                    'RESULT': str
                }
            )
            
            # Mostrar información sobre las columnas encontradas
            st.info(f"Columnas encontradas en el archivo: {', '.join(df.columns)}")
            
            # Verificar y renombrar columnas si es necesario
            column_mapping = {
                'RSID': 'RSID',
                'CHROMOSOME': 'CHROMOSOME',
                'POSITION': 'POSITION',
                'RESULT': 'RESULT',
                'GENOTYPE': 'RESULT'  # También aceptar GENOTYPE como nombre alternativo
            }
            
            # Renombrar columnas si es necesario
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and old_col != new_col:
                    df = df.rename(columns={old_col: new_col})
                    st.info(f"Columna renombrada: {old_col} -> {new_col}")
            
            # Mostrar vista previa de los datos
            st.subheader("Vista previa de los datos")
            st.dataframe(df.head())
            
            # Validar los datos
            if validate_myheritage_data(df):
                # Convertir el archivo
                with st.spinner("🔄 Convirtiendo archivo MyHeritage al formato requerido..."):
                    df_convertido, session_id = convert_myheritage(df)
                
                if df_convertido is not None:
                    # Mostrar vista previa de los datos convertidos
                    st.success("✅ Archivo convertido correctamente")
                    st.subheader("Vista previa de los datos convertidos")
                    st.dataframe(df_convertido.head())
                    
                    # Procesar los datos y generar el informe
                    with st.spinner("🔄 Procesando datos, espera por favor..."):
                        if procesar_datos_geneticos(df_convertido, session_id):
                            st.success("✅ Procesamiento completado exitosamente")
                        else:
                            st.error("❌ Hubo un error al procesar tus datos. Por favor, intenta nuevamente.")
                else:
                    st.error("❌ No se pudo convertir el archivo. Por favor, verifica el formato y vuelve a intentarlo.")
            else:
                st.error("❌ El archivo no cumple con los requisitos de formato. Por favor, verifica las columnas y los datos.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Por favor, verifica que el archivo tenga el formato correcto y vuelve a intentarlo.")

# =============================================================================
# Ejecución de la Aplicación
# =============================================================================

if __name__ == "__main__":
    main()

optimizer = DatabaseOptimizer()
optimizer.create_indexes() 