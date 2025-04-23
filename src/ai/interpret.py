#!/usr/bin/env python3
"""
Interpretación de Resultados PRS usando OpenAI GPT-3.5-turbo

Este script utiliza la API de OpenAI para generar interpretaciones naturales y comprensibles
de los resultados de Polygenic Risk Scores (PRS) para usuarios no técnicos.

CONFIGURACIÓN REQUERIDA:
1. Instalar la librería de OpenAI: pip install openai
2. Configurar la API key de OpenAI:
   - Crear un archivo .env en la raíz del proyecto
   - Agregar: OPENAI_API_KEY=tu-api-key-aquí
   - O exportar la variable: export OPENAI_API_KEY=tu-api-key-aquí
"""

import os
import sys
import logging
from typing import Optional
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Intentar importar openai con manejo de error
try:
    import openai
except ImportError:
    logger.error("La librería 'openai' no está instalada. Por favor, ejecuta: pip install openai")
    sys.exit(1)

def configurar_openai() -> None:
    """
    Configura la API key de OpenAI desde variables de entorno.
    
    Raises:
        ValueError: Si no se encuentra la API key
    """
    # Cargar variables de entorno
    load_dotenv()
    
    # Obtener API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key de OpenAI no encontrada. "
            "Por favor, configura OPENAI_API_KEY en el archivo .env o como variable de entorno."
        )
    
    # Configurar OpenAI
    openai.api_key = api_key
    logger.info("API de OpenAI configurada correctamente")

def generar_interpretacion(resultado_prs: str) -> Optional[str]:
    """
    Genera una interpretación natural de los resultados PRS usando GPT-3.5-turbo.
    
    Args:
        resultado_prs: String con los resultados del PRS a interpretar
        
    Returns:
        str: Interpretación generada o None en caso de error
        
    Raises:
        Exception: Si hay problemas con la API de OpenAI
    """
    try:
        # Crear prompt para GPT-3.5-turbo
        prompt = f"""
        Por favor, genera una interpretación clara y natural de los siguientes resultados de riesgo genético.
        La interpretación debe ser:
        - Breve (máximo 2 párrafos)
        - Fácil de entender para personas sin conocimientos técnicos
        - Educativa y orientativa
        - Basada en evidencia científica
        - Enfocada en acciones prácticas
        
        Resultados a interpretar: {resultado_prs}
        
        Por favor, evita:
        - Términos técnicos complejos
        - Alarmismo innecesario
        - Recomendaciones médicas específicas
        - Comparaciones con otros individuos
        """
        
        # Llamar a la API de OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asesor genético experto que explica resultados de riesgo genético de manera clara y compasiva."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Extraer y retornar la interpretación
        interpretacion = response.choices[0].message.content.strip()
        return interpretacion
    
    except openai.error.AuthenticationError:
        logger.error("Error de autenticación con OpenAI. Verifica tu API key.")
        return None
    except openai.error.APIError as e:
        logger.error(f"Error en la API de OpenAI: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return None

def main():
    """Función principal para ejecutar la interpretación."""
    try:
        # Configurar OpenAI
        configurar_openai()
        
        # Ejemplo de resultado PRS (en producción, esto vendría de calculate_prs.py)
        resultado_ejemplo = "Tu puntaje PRS para Diabetes tipo 2 es 2.8, indicando un riesgo moderadamente alto comparado con la población general."
        
        # Generar interpretación
        logger.info("Generando interpretación...")
        interpretacion = generar_interpretacion(resultado_ejemplo)
        
        if interpretacion:
            print("\nInterpretación de Resultados")
            print("=========================")
            print(interpretacion)
            print("\nNota: Esta interpretación es generada por IA y debe ser considerada como información general.")
            print("Para asesoramiento médico específico, consulta con un profesional de la salud.")
        else:
            print("No se pudo generar la interpretación. Por favor, revisa los logs para más detalles.")
    
    except Exception as e:
        logger.error(f"Error en la ejecución: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 