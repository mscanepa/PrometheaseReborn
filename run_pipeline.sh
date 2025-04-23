#!/bin/bash

# TheModernPromethease Pipeline Script
# This script orchestrates the execution of the genomic analysis pipeline

# Configuración de logging
LOG_FILE=${LOG_FILE:-"pipeline.log"}
echo "Iniciando pipeline de análisis genético - $(date)" > "$LOG_FILE"

# Función para logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Función para manejar errores
error() {
    log "ERROR: $1"
    exit 1
}

# Verificar dependencias
log "Verificando dependencias..."
command -v python3 >/dev/null 2>&1 || error "Python3 no está instalado"
command -v pip >/dev/null 2>&1 || error "pip no está instalado"

# Verificar archivos necesarios
log "Verificando archivos necesarios..."
[ -f "src/extract_data.py" ] || error "No se encontró extract_data.py"
[ -f "src/calculate_prs.py" ] || error "No se encontró calculate_prs.py"
[ -f "src/generate_report.py" ] || error "No se encontró generate_report.py"
[ -f "src/ai/interpret.py" ] || error "No se encontró interpret.py"

# Verificar archivo de datos
log "Verificando archivo de datos..."
[ -f "data/raw/usuario_adaptado.txt" ] || error "No se encontró el archivo de datos"

# Verificar permisos de directorios
log "Verificando permisos de directorios..."
[ -w "data/processed" ] || error "No hay permisos de escritura en data/processed"
[ -w "reports" ] || error "No hay permisos de escritura en reports"

# Ejecutar cada paso del pipeline con logging
log "Iniciando extracción de datos..."
python3 src/extract_data.py >> "$LOG_FILE" 2>&1 || error "Error en extract_data.py"

log "Calculando PRS..."
python3 src/calculate_prs.py >> "$LOG_FILE" 2>&1 || error "Error en calculate_prs.py"

log "Generando reporte..."
python3 src/generate_report.py >> "$LOG_FILE" 2>&1 || error "Error en generate_report.py"

log "Generando interpretación con IA..."
python3 src/ai/interpret.py >> "$LOG_FILE" 2>&1 || error "Error en interpret.py"

# Set error handling
set -e
set -o pipefail

# Configure logging
log_file="pipeline.log"
exec 1> >(tee -a "$log_file")
exec 2>&1

echo "Starting TheModernPromethease Pipeline"
echo "====================================="
echo "Timestamp: $(date)"
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/raw data/processed reports

# Step 1: Data Extraction
echo ""
echo "Step 1: Data Extraction"
echo "----------------------"
python src/extract_data.py
if [ $? -ne 0 ]; then
    echo "Error in data extraction step"
    exit 1
fi

# Step 2: PRS Calculation
echo ""
echo "Step 2: PRS Calculation"
echo "----------------------"
python src/calculate_prs.py
if [ $? -ne 0 ]; then
    echo "Error in PRS calculation step"
    exit 1
fi

# Step 3: Report Generation
echo ""
echo "Step 3: Report Generation"
echo "----------------------"
python src/generate_report.py
if [ $? -ne 0 ]; then
    echo "Error in report generation step"
    exit 1
fi

# Step 4: AI Interpretation
echo ""
echo "Step 4: AI Interpretation"
echo "----------------------"
python src/ai/interpret.py
if [ $? -ne 0 ]; then
    echo "Error in AI interpretation step"
    exit 1
fi

echo ""
echo "Pipeline completed successfully!"
echo "Timestamp: $(date)"
echo "Log file: $log_file" 