#!/usr/bin/env python3
"""
Convertidor de archivos MyHeritage a formato TheModernPromethease

Este script convierte archivos CSV de MyHeritage al formato requerido por la aplicación.
El formato de salida tendrá las columnas: rsid, chromosome, position, genotype
"""

import pandas as pd
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convertir_archivo(archivo_entrada, archivo_salida):
    """
    Convierte un archivo CSV de MyHeritage al formato requerido.
    
    Args:
        archivo_entrada: Ruta al archivo CSV de MyHeritage
        archivo_salida: Ruta donde se guardará el archivo convertido
    """
    try:
        # Leer el archivo CSV
        logger.info(f"Leyendo archivo: {archivo_entrada}")
        df = pd.read_csv(archivo_entrada)
        
        # Verificar columnas requeridas
        columnas_requeridas = ['rsid', 'chromosome', 'position', 'genotype']
        if not all(col in df.columns for col in columnas_requeridas):
            # Si no tiene las columnas correctas, intentar mapear las de MyHeritage
            mapeo_columnas = {
                'RSID': 'rsid',
                'CHROMOSOME': 'chromosome',
                'POSITION': 'position',
                'RESULT': 'genotype'
            }
            
            # Renombrar columnas según el mapeo
            for col_original, col_nueva in mapeo_columnas.items():
                if col_original in df.columns:
                    df = df.rename(columns={col_original: col_nueva})
            
            # Verificar si tenemos todas las columnas necesarias
            if not all(col in df.columns for col in columnas_requeridas):
                raise ValueError("El archivo no contiene las columnas necesarias")
        
        # Seleccionar y ordenar columnas
        df = df[columnas_requeridas]
        
        # Guardar en formato de texto con tabulaciones
        logger.info(f"Guardando archivo convertido: {archivo_salida}")
        df.to_csv(archivo_salida, sep='\t', index=False)
        
        logger.info("Conversión completada exitosamente")
        return True
    
    except Exception as e:
        logger.error(f"Error durante la conversión: {str(e)}")
        return False

def main():
    """Función principal para ejecutar la conversión."""
    if len(sys.argv) != 3:
        print("Uso: python convert_myheritage.py <archivo_entrada.csv> <archivo_salida.txt>")
        sys.exit(1)
    
    archivo_entrada = sys.argv[1]
    archivo_salida = sys.argv[2]
    
    if convertir_archivo(archivo_entrada, archivo_salida):
        print(f"\nArchivo convertido exitosamente.")
        print(f"Archivo de salida: {archivo_salida}")
        print("\nAhora puedes usar este archivo en la aplicación web.")
    else:
        print("\nError durante la conversión. Por favor, verifica el archivo de entrada.")

if __name__ == "__main__":
    main() 