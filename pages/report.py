import streamlit as st
import pandas as pd
import os
from datetime import datetime
import logging
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Configurar Pandas para manejar DataFrames grandes
pd.set_option("styler.render.max_elements", 2000000)  # Aumentar el límite de elementos para styling
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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

# Configuración de la página - DEBE SER LA PRIMERA LLAMADA A STREAMLIT
st.set_page_config(
    page_title="TheModernPromethease - Reporte",
    page_icon="📊",
    layout="wide"
)

def is_matplotlib_available():
    """Verifica si matplotlib está disponible."""
    try:
        import matplotlib
        return True
    except ImportError:
        return False

def load_report_data(session_id):
    """Carga los datos del reporte para una sesión específica."""
    try:
        # Buscar el archivo de reporte que coincida con el session_id
        report_files = [f for f in os.listdir("reports") if f.startswith("report_") and f.endswith(".parquet")]
        matching_files = [f for f in report_files if session_id in f]
        
        if not matching_files:
            logger.error(f"No se encontró el reporte para la sesión {session_id}")
            st.error(f"❌ No se encontró el reporte para la sesión {session_id}")
            st.info("Los reportes disponibles son:")
            for file in report_files:
                session = file.replace("report_", "").replace(".parquet", "")
                st.code(session, language="text")
            return None
        
        if len(matching_files) > 1:
            logger.warning(f"Se encontraron múltiples reportes para la sesión {session_id}")
            st.warning("⚠️ Se encontraron múltiples reportes. Usando el más reciente.")
            
        # Usar el archivo más reciente si hay múltiples coincidencias
        report_path = os.path.join("reports", sorted(matching_files)[-1])
        logger.info(f"Cargando reporte desde: {report_path}")
        
        return pd.read_parquet(report_path)
    except Exception as e:
        logger.error(f"Error al cargar el reporte: {str(e)}")
        st.error(f"❌ Error al cargar el reporte: {str(e)}")
        return None

def load_myheritage_data(session_id):
    """Carga los datos de MyHeritage para la sesión específica."""
    try:
        logger.info(f"Session ID received: {session_id}")
        myheritage_path = os.path.join("data", "myheritage", f"myheritage_{session_id}.json")
        logger.info(f"Constructed file path: {myheritage_path}")
        logger.info(f"File exists: {os.path.exists(myheritage_path)}")
        
        # Si el archivo no existe, intentar buscar otros archivos que puedan coincidir
        if not os.path.exists(myheritage_path):
            logger.info("File not found, searching for matching MyHeritage files...")
            myheritage_dir = os.path.join("data", "myheritage")
            myheritage_files = [f for f in os.listdir(myheritage_dir) if f.startswith("myheritage_") and f.endswith(".json")]
            
            # Buscar archivos que contengan el session_id (para IDs parciales)
            matching_files = [f for f in myheritage_files if session_id in f.replace("myheritage_", "").replace(".json", "")]
            
            if matching_files:
                # Usar el archivo más reciente si hay múltiples coincidencias
                selected_file = sorted(matching_files)[-1]
                myheritage_path = os.path.join(myheritage_dir, selected_file)
                logger.info(f"Found matching MyHeritage file: {myheritage_path}")
            else:
                logger.warning(f"No se encontraron datos de MyHeritage para la sesión {session_id}")
                return None
            
        with open(myheritage_path, 'r') as f:
            myheritage_data = json.load(f)
            
        # Convertir los datos a DataFrame
        variants_data = []
        for variant in myheritage_data.get('variants', []):
            variants_data.append({
                'rsid': variant.get('rsid'),
                'chromosome': variant.get('chromosome'),
                'position': variant.get('position'),
                'genotype': variant.get('genotype'),
                'source': 'MyHeritage'
            })
            
        if not variants_data:
            return None
            
        return pd.DataFrame(variants_data)
    except Exception as e:
        logger.error(f"Error al cargar datos de MyHeritage: {str(e)}")
        return None

def merge_with_myheritage(df, myheritage_df):
    """Combina los datos del reporte con los datos de MyHeritage."""
    if myheritage_df is None:
        return df
        
    # Crear columnas para datos de MyHeritage
    df['found_in_myheritage'] = False
    df['myheritage_genotype'] = None
    df['myheritage_chromosome'] = None
    df['myheritage_position'] = None
    
    # Mapear los datos de MyHeritage
    myheritage_map = myheritage_df.set_index('rsid').to_dict('index')
    
    # Actualizar el DataFrame con los datos de MyHeritage
    for idx, row in df.iterrows():
        rsid = row.get('rsid')
        if rsid in myheritage_map:
            df.at[idx, 'found_in_myheritage'] = True
            df.at[idx, 'myheritage_genotype'] = myheritage_map[rsid]['genotype']
            df.at[idx, 'myheritage_chromosome'] = myheritage_map[rsid]['chromosome']
            df.at[idx, 'myheritage_position'] = myheritage_map[rsid]['position']
    
    return df

def display_report(df):
    """Muestra el reporte de manera interactiva."""
    if df is None:
        return
    
    # Eliminar columnas completamente vacías
    df_cleaned = df.dropna(axis=1, how='all')
    if len(df.columns) != len(df_cleaned.columns):
        removed_cols = set(df.columns) - set(df_cleaned.columns)
        st.info(f"📝 Se ocultaron {len(removed_cols)} columnas completamente vacías: {', '.join(removed_cols)}")
        df = df_cleaned

    # Filtrar solo variantes de MyHeritage
    if 'found_in_myheritage' in df.columns:
        df = df[df['found_in_myheritage']]
        if len(df) == 0:
            st.error("❌ No se encontraron variantes en tus datos de MyHeritage")
            return
    else:
        st.error("❌ No hay datos de MyHeritage disponibles")
        return
    
    # Extraer información de genes si está disponible
    if 'clinvar_phenotype' in df.columns and 'gene_name' not in df.columns:
        # Intentar extraer el nombre del gen del campo clinvar_phenotype
        df['gene_name'] = df['clinvar_phenotype'].astype(str).str.extract(r'gene=([A-Za-z0-9]+)', expand=False)
        logger.info("Extracted gene names from clinvar_phenotype field")
        
    # Intentar identificar genes para variantes conocidas
    df = identify_potential_genes(df)
        
    # Mapear los nombres de columnas a los nombres esperados
    column_mapping = {
        'clinvar_clinical_significance': 'clinical_significance',
        'gwas_disease_trait': 'gwas_trait',
        'dbsnp_function': 'dbsnp_function',
        'clinvar_phenotype': 'phenotype',
        'gene_symbol': 'gene_name'
    }
    
    # Renombrar columnas según el mapeo
    for original_col, new_col in column_mapping.items():
        if original_col in df.columns and new_col not in df.columns:
            df[new_col] = df[original_col]
            logger.info(f"Mapped column {original_col} to {new_col}")

    # Verificar columnas requeridas
    required_columns = ['clinical_significance', 'gwas_trait', 'dbsnp_function']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"⚠️ Las siguientes columnas están ausentes en los datos: {', '.join(missing_columns)}")
        # Agregar columnas faltantes con valores NaN
        for col in missing_columns:
            df[col] = pd.NA
    
    # Clasificar variantes
    mapped_variants = df[
        df['clinical_significance'].notna() |
        df['gwas_trait'].notna() |
        df['dbsnp_function'].notna()
    ].copy()
    
    unmapped_variants = df[
        df['clinical_significance'].isna() &
        df['gwas_trait'].isna() &
        df['dbsnp_function'].isna()
    ].copy()

    # Resumen General
    st.header("🧬 Resumen de Tu Análisis Genético")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Variantes", len(df))
    with col2:
        st.metric("Variantes Mapeadas", len(mapped_variants))
    with col3:
        st.metric("Variantes Sin Mapear", len(unmapped_variants))
    with col4:
        genes_count = df['gene_name'].notna().sum() if 'gene_name' in df.columns else 0
        st.metric("Genes Identificados", genes_count)

    # Vista por Genes
    st.header("🧬 Análisis por Genes")
    if 'gene_name' in df.columns and df['gene_name'].notna().any():
        # Agrupar variantes por gen
        gene_variants = df[df['gene_name'].notna()].groupby('gene_name').agg({
            'rsid': 'count',
            'clinical_significance': lambda x: x.notna().sum() if 'clinical_significance' in df.columns else 0,
            'gwas_trait': lambda x: x.notna().sum() if 'gwas_trait' in df.columns else 0,
            'dbsnp_function': lambda x: x.notna().sum() if 'dbsnp_function' in df.columns else 0
        }).reset_index()
        
        gene_variants.columns = ['Gen', 'Total Variantes', 'Con Info Clínica', 'Con Estudios GWAS', 'Con Info dbSNP']
        
        # Filtro de genes
        selected_gene = st.selectbox(
            "🔍 Selecciona un gen para ver detalles",
            options=['Todos los genes'] + sorted(gene_variants['Gen'].unique().tolist())
        )
        
        if selected_gene != 'Todos los genes':
            # Mostrar detalles del gen seleccionado
            gene_data = df[df['gene_name'] == selected_gene]
            
            # Métricas del gen
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Variantes en el Gen", len(gene_data))
            with col2:
                clinical_variants = gene_data['clinical_significance'].notna().sum() if 'clinical_significance' in gene_data.columns else 0
                st.metric("Variantes con Significancia Clínica", clinical_variants)
            with col3:
                gwas_variants = gene_data['gwas_trait'].notna().sum() if 'gwas_trait' in gene_data.columns else 0
                st.metric("Variantes con Estudios GWAS", gwas_variants)
            
            # Mostrar variantes del gen
            st.subheader(f"Variantes en el Gen {selected_gene}")
            for _, variant in gene_data.iterrows():
                with st.expander(f"📍 {variant.get('rsid', 'N/A')} - {variant.get('myheritage_genotype', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Información Básica**")
                        st.write(f"Cromosoma: {variant.get('myheritage_chromosome', 'N/A')}")
                        st.write(f"Posición: {variant.get('myheritage_position', 'N/A')}")
                        st.write(f"Genotipo: {variant.get('myheritage_genotype', 'N/A')}")
                    
                    with col2:
                        st.write("**Información Clínica**")
                        if 'clinical_significance' in variant and pd.notna(variant.get('clinical_significance')):
                            st.write(f"Significancia: {variant['clinical_significance']}")
                        if 'gwas_trait' in variant and pd.notna(variant.get('gwas_trait')):
                            st.write(f"Estudios GWAS: {variant['gwas_trait']}")
                        if 'dbsnp_function' in variant and pd.notna(variant.get('dbsnp_function')):
                            st.write(f"Función: {variant['dbsnp_function']}")
        else:
            # Mostrar resumen de todos los genes
            try:
                if is_matplotlib_available():
                    st.dataframe(gene_variants.style.background_gradient(cmap='YlOrRd', subset=['Total Variantes']))
                else:
                    st.dataframe(gene_variants)
                    st.info("Para visualizar con gradientes de color, instala matplotlib: pip install matplotlib")
            except Exception as e:
                st.dataframe(gene_variants)
                st.warning(f"No se pudo aplicar estilo al dataframe: {str(e)}")
            
            # Gráfico de distribución de variantes por gen
            try:
                fig = px.bar(gene_variants.head(20), 
                           x='Gen', 
                           y='Total Variantes',
                           title='Top 20 Genes por Número de Variantes')
                st.plotly_chart(fig)
            except Exception as e:
                st.warning(f"No se pudo generar el gráfico: {str(e)}")
    else:
        st.info("No hay información de genes disponible en los datos.")

    # Distribución de Significancia Clínica
    st.header("📊 Distribución de Significancia Clínica")
    if 'clinical_significance' in mapped_variants.columns and mapped_variants['clinical_significance'].notna().any():
        try:
            clinical_counts = mapped_variants['clinical_significance'].value_counts()
            
            # Gráfico de pastel para significancia clínica
            fig = go.Figure(data=[go.Pie(
                labels=clinical_counts.index,
                values=clinical_counts.values,
                hole=.3
            )])
            fig.update_layout(title='Distribución de Significancia Clínica')
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de significancia clínica: {str(e)}")
            # Mostrar una tabla como alternativa
            st.write("Distribución de Significancia Clínica (tabla):")
            clinical_counts = mapped_variants['clinical_significance'].value_counts().reset_index()
            clinical_counts.columns = ['Significancia Clínica', 'Cantidad']
            st.dataframe(clinical_counts)
    else:
        st.info("No hay información de significancia clínica disponible en los datos.")

    # Filtros en la barra lateral
    st.sidebar.header("🔍 Filtros")
    
    # Filtro por fuente de datos
    data_sources = []
    if 'clinical_significance' in mapped_variants.columns and mapped_variants['clinical_significance'].notna().any():
        data_sources.append('ClinVar')
    if 'gwas_trait' in mapped_variants.columns and mapped_variants['gwas_trait'].notna().any():
        data_sources.append('GWAS')
    if 'dbsnp_function' in mapped_variants.columns and mapped_variants['dbsnp_function'].notna().any():
        data_sources.append('dbSNP')
    
    selected_sources = st.sidebar.multiselect(
        "🗃️ Fuentes de Datos",
        data_sources
    )

    # Filtro por significancia clínica
    selected_significance = []
    if 'clinical_significance' in mapped_variants.columns and mapped_variants['clinical_significance'].notna().any():
        significance_options = sorted(mapped_variants['clinical_significance'].dropna().unique())
        selected_significance = st.sidebar.multiselect(
            "🏥 Significancia Clínica",
            significance_options,
            default=[sig for sig in significance_options if 'pathogenic' in str(sig).lower()]
        )

    # Aplicar filtros
    filtered_variants = mapped_variants.copy()
    if selected_sources:
        mask = pd.Series(False, index=filtered_variants.index)
        if 'ClinVar' in selected_sources and 'clinical_significance' in filtered_variants.columns:
            mask |= filtered_variants['clinical_significance'].notna()
        if 'GWAS' in selected_sources and 'gwas_trait' in filtered_variants.columns:
            mask |= filtered_variants['gwas_trait'].notna()
        if 'dbSNP' in selected_sources and 'dbsnp_function' in filtered_variants.columns:
            mask |= filtered_variants['dbsnp_function'].notna()
        filtered_variants = filtered_variants[mask]

    if selected_significance and 'clinical_significance' in filtered_variants.columns:
        filtered_variants = filtered_variants[
            filtered_variants['clinical_significance'].isin(selected_significance)
        ]

    # Mostrar variantes filtradas
    try:
        if not filtered_variants.empty:
            st.header("🔍 Variantes Filtradas")
            st.write(f"Mostrando {len(filtered_variants)} variantes que cumplen los criterios seleccionados")
            
            # Opciones de visualización
            all_columns = filtered_variants.columns.tolist()
            default_columns = [col for col in ['rsid', 'gene_name', 'clinical_significance', 'gwas_trait'] if col in all_columns]
            
            display_options = st.multiselect(
                "Columnas a mostrar",
                all_columns,
                default=default_columns
            )
            
            if display_options:
                # Seleccionar solo las columnas que existen
                valid_columns = [col for col in display_options if col in filtered_variants.columns]
                if valid_columns:
                    st.dataframe(filtered_variants[valid_columns])
                else:
                    st.warning("No se pudieron mostrar las columnas seleccionadas.")
                
            # Opción de descarga
            try:
                csv = filtered_variants.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Variantes Filtradas (CSV)",
                    csv,
                    "variantes_filtradas.csv",
                    "text/csv",
                    key='download-csv'
                )
            except Exception as e:
                st.error(f"Error al preparar el archivo para descarga: {str(e)}")
        elif filtered_variants.empty and (selected_sources or selected_significance):
            st.info("No se encontraron variantes que cumplan los criterios de filtrado seleccionados.")
    except Exception as e:
        st.error(f"Error al mostrar las variantes filtradas: {str(e)}")
        logger.error(f"Error en el display de variantes filtradas: {str(e)}")

def get_report_metadata(session_id):
    """Obtiene los metadatos del reporte."""
    try:
        metadata_path = os.path.join("reports", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return metadata.get(session_id, {})
    except Exception as e:
        logger.error(f"Error al cargar metadatos: {str(e)}")
        return {}

def save_report_metadata(session_id, metadata):
    """Guarda los metadatos del reporte."""
    try:
        metadata_path = os.path.join("reports", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        all_metadata[session_id] = metadata
        
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=4)
            
        return True
    except Exception as e:
        logger.error(f"Error al guardar metadatos: {str(e)}")
        return False

def list_available_reports():
    """Lista todos los reportes disponibles."""
    reports = []
    try:
        # Obtener todos los archivos de reporte
        report_files = [f for f in os.listdir("reports") if f.startswith("report_") and f.endswith(".parquet")]
        metadata = get_all_metadata()
        
        for report_file in report_files:
            session_id = report_file.replace("report_", "").replace(".parquet", "")
            myheritage_path = os.path.join("data", "myheritage", f"myheritage_{session_id}.json")
            report_info = {
                'session_id': session_id,
                'file_name': report_file,
                'created_at': datetime.fromtimestamp(os.path.getctime(os.path.join("reports", report_file))).strftime('%Y-%m-%d %H:%M:%S'),
                'size': os.path.getsize(os.path.join("reports", report_file)) / (1024 * 1024),  # Tamaño en MB
                'name': metadata.get(session_id, {}).get('name', 'Sin nombre'),
                'description': metadata.get(session_id, {}).get('description', ''),
                'has_myheritage': os.path.exists(myheritage_path)
            }
            reports.append(report_info)
        
        # Ordenar por fecha de creación, más reciente primero
        reports.sort(key=lambda x: x['created_at'], reverse=True)
        
    except Exception as e:
        logger.error(f"Error al listar reportes: {str(e)}")
    
    return reports

def get_all_metadata():
    """Obtiene todos los metadatos."""
    try:
        metadata_path = os.path.join("reports", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error al cargar todos los metadatos: {str(e)}")
        return {}

def display_report_list():
    """Muestra la lista de reportes disponibles."""
    st.header("📋 Reportes Disponibles")
    
    reports = list_available_reports()
    if not reports:
        st.warning("No se encontraron reportes")
        return
    
    # Crear una tabla con los reportes
    report_data = []
    for report in reports:
        # Crear el enlace HTML
        report_link = f'<a href="?id={report["session_id"]}" target="_self">📊 Ver Reporte</a>'
        
        report_data.append({
            "Nombre": report['name'],
            "ID de Sesión": report['session_id'],
            "Fecha de Creación": report['created_at'],
            "Tamaño (MB)": f"{report['size']:.2f}",
            "MyHeritage": "✅" if report['has_myheritage'] else "❌",
            "Descripción": report['description'] or "Sin descripción",
            "Acciones": report_link
        })
    
    # Convertir a DataFrame
    df = pd.DataFrame(report_data)
    
    # Mostrar la tabla con enlaces HTML habilitados
    st.markdown(
        df.to_html(
            escape=False,
            index=False,
            table_id="reports_table"
        ),
        unsafe_allow_html=True
    )
    
    # Agregar estilo CSS para mejorar la apariencia de la tabla
    st.markdown("""
        <style>
            #reports_table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-size: 14px;
            }
            #reports_table th {
                background-color: #f0f2f6;
                padding: 12px;
                text-align: left;
                border-bottom: 2px solid #ddd;
            }
            #reports_table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            #reports_table tr:hover {
                background-color: #f5f5f5;
            }
            #reports_table a {
                text-decoration: none;
                color: #ff4b4b;
                font-weight: 500;
                padding: 5px 10px;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            #reports_table a:hover {
                background-color: #ff4b4b;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

def identify_potential_genes(df):
    """Intenta identificar genes potenciales a partir de las variantes disponibles."""
    try:
        # Si ya tenemos información de genes, no es necesario
        if 'gene_name' in df.columns and df['gene_name'].notna().any():
            return df
            
        # Crear una columna para almacenar los genes potenciales
        df['gene_name'] = pd.NA
        
        # Intentar asignar genes basados en el rsid
        # Esta es una aproximación, idealmente se consultaría una base de datos genómica
        common_gene_variants = {
            'rs429358': 'APOE',
            'rs7412': 'APOE',
            'rs1800562': 'HFE',
            'rs1801133': 'MTHFR',
            'rs1799945': 'HFE',
            'rs1051730': 'CHRNA3',
            'rs8177374': 'TIRAP',
            'rs3135506': 'APOA5',
            'rs662': 'PON1',
            'rs1800497': 'ANKK1',
            'rs53576': 'OXTR',
            'rs2295583': 'SMAD3',
            'rs17822931': 'ABCC11',
            'rs4680': 'COMT',
            'rs1815739': 'ACTN3',
            'rs1799971': 'OPRM1',
            'rs713598': 'TAS2R38',
            'rs1726866': 'TAS2R38'
        }
        
        # Asignar los genes conocidos
        for rsid, gene in common_gene_variants.items():
            mask = df['rsid'] == rsid
            if mask.any():
                df.loc[mask, 'gene_name'] = gene
                
        logger.info(f"Assigned {df['gene_name'].notna().sum()} variants to known genes")
        return df
        
    except Exception as e:
        logger.error(f"Error identifying potential genes: {str(e)}")
        return df

def main():
    """Función principal de la página de reporte."""
    st.title("📊 Reporte de Análisis Genético")
    
    # Obtener el ID de sesión de los parámetros de la URL
    params = st.query_params
    session_id = params.get("id", [None])[0]

    logger.info(f"Extracted session ID from URL: {session_id}")

    # Si no hay ID de sesión, mostrar la lista de reportes
    if not session_id:
        st.warning("❌ No se proporcionó un ID de sesión válido")
        st.info("Por favor, selecciona un reporte de la lista:")
        display_report_list()
        return

    # Si el ID de sesión no contiene guiones bajos, podría ser un ID corto para retrocompatibilidad
    if session_id and session_id.isdigit():
        logger.info(f"Numeric session ID detected: {session_id}. Searching for matching report files...")
        # Buscar archivos de reporte que puedan coincidir con este ID corto
        report_files = [f for f in os.listdir("reports") if f.startswith("report_") and f.endswith(".parquet")]
        matching_files = [f for f in report_files if session_id in f.replace("report_", "").replace(".parquet", "")]
        
        if matching_files:
            # Usar el ID completo del archivo más reciente
            full_session_id = sorted(matching_files)[-1].replace("report_", "").replace(".parquet", "")
            logger.info(f"Found full session ID: {full_session_id} for short ID: {session_id}")
            session_id = full_session_id

    # Cargar metadatos del reporte
    metadata = get_report_metadata(session_id)
    
    # Sección de información del reporte
    with st.expander("ℹ️ Información del Reporte", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            report_name = st.text_input(
                "Nombre del Reporte",
                value=metadata.get('name', ''),
                key="report_name"
            )
        
        with col2:
            report_description = st.text_area(
                "Descripción",
                value=metadata.get('description', ''),
                key="report_description",
                height=100
            )
        
        if st.button("💾 Guardar Información"):
            metadata = {
                'name': report_name,
                'description': report_description,
                'last_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            if save_report_metadata(session_id, metadata):
                st.success("✅ Información guardada exitosamente")
            else:
                st.error("❌ Error al guardar la información")
    
    st.info(f"ID de Sesión: {session_id}")
    
    # Cargar datos
    df = load_report_data(session_id)
    if df is not None:
        # Cargar y combinar datos de MyHeritage
        myheritage_df = load_myheritage_data(session_id)
        if myheritage_df is not None:
            df = merge_with_myheritage(df, myheritage_df)
            st.success("✅ Datos de MyHeritage cargados y combinados exitosamente")
        else:
            st.warning("⚠️ No se encontraron datos de MyHeritage para esta sesión")
        
        # Mostrar el reporte
        display_report(df)

if __name__ == "__main__":
    main() 