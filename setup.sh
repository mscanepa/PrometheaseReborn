#!/bin/bash

# Script de inicialización para TheModernPromethease
# Verifica requisitos y configura el entorno

# Colores para la salida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Iniciando configuración de TheModernPromethease...${NC}\n"

# Verificar Python
echo "Verificando Python..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION instalado${NC}"
    else
        echo -e "${RED}✗ Se requiere Python 3.8 o superior${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python no encontrado${NC}"
    exit 1
fi

# Verificar pip
echo "Verificando pip..."
if command -v pip3 &>/dev/null; then
    echo -e "${GREEN}✓ pip instalado${NC}"
else
    echo -e "${RED}✗ pip no encontrado${NC}"
    exit 1
fi

# Crear directorios necesarios
echo "Creando estructura de directorios..."
DIRS=("data/raw" "data/processed" "reports")
for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✓ Directorio $dir creado${NC}"
    else
        echo -e "${YELLOW}✓ Directorio $dir ya existe${NC}"
    fi
done

# Verificar archivo .env
echo "Verificando configuración de OpenAI..."
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ Archivo .env no encontrado${NC}"
    echo "Por favor, crea un archivo .env con tu API key de OpenAI:"
    echo "OPENAI_API_KEY=tu-api-key"
    exit 1
else
    if grep -q "OPENAI_API_KEY" .env; then
        echo -e "${GREEN}✓ Configuración de OpenAI encontrada${NC}"
    else
        echo -e "${RED}✗ OPENAI_API_KEY no configurada en .env${NC}"
        exit 1
    fi
fi

# Verificar dependencias
echo "Verificando dependencias..."
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias..."
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencias instaladas correctamente${NC}"
    else
        echo -e "${RED}✗ Error al instalar dependencias${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Archivo requirements.txt no encontrado${NC}"
    exit 1
fi

# Hacer ejecutable el script run_pipeline.sh
if [ -f "run_pipeline.sh" ]; then
    chmod +x run_pipeline.sh
    echo -e "${GREEN}✓ Script run_pipeline.sh hecho ejecutable${NC}"
fi

echo -e "\n${GREEN}Configuración completada exitosamente!${NC}"
echo -e "Puedes comenzar a usar TheModernPromethease ejecutando:"
echo -e "${YELLOW}./run_pipeline.sh${NC}\n" 