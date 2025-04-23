# TheModernPromethease

TheModernPromethease es una herramienta avanzada para el análisis de riesgo poligénico y la interpretación de datos genómicos. El proyecto integra análisis estadísticos con interpretación asistida por IA para proporcionar insights valiosos sobre predisposiciones genéticas.

## Estructura del Proyecto

```
TheModernPromethease/
├── src/
│   ├── extract_data.py      # Extracción y procesamiento de datos
│   ├── calculate_prs.py     # Cálculo de puntajes de riesgo poligénico
│   ├── generate_report.py   # Generación de reportes y visualizaciones
│   └── ai/
│       └── interpret.py     # Interpretación AI de resultados
├── data/
│   ├── raw/                 # Datos sin procesar
│   └── processed/           # Datos procesados
├── reports/                 # Reportes generados
├── requirements.txt         # Dependencias del proyecto
└── run_pipeline.sh          # Script de ejecución del pipeline
```

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Cuenta de OpenAI con API key válida
- Acceso a datos genómicos en formato compatible (23andMe, MyHeritage)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/TheModernPromethease.git
cd TheModernPromethease
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
echo "OPENAI_API_KEY=tu-api-key" > .env
```

## Uso

1. Preparar datos:
   - Colocar archivo de datos genómicos en `data/raw/usuario_adaptado.txt`
   - Asegurar que los datos GWAS estén en `data/processed/gwas_catalog_processed.csv`

2. Ejecutar el pipeline completo:
```bash
./run_pipeline.sh
```

3. Ver resultados:
   - Los reportes se generan en el directorio `reports/`
   - Cada ejecución crea un nuevo directorio con timestamp
   - Incluye visualizaciones y reportes HTML

## Componentes Principales

### Extracción de Datos
- Procesamiento de datos genómicos
- Integración con bases de datos GWAS
- Limpieza y normalización de datos

### Cálculo de PRS
- Implementación de algoritmos de riesgo poligénico
- Normalización de puntajes
- Análisis estadístico

### Generación de Reportes
- Visualizaciones interactivas
- Reportes HTML detallados
- Exportación de resultados

### Interpretación AI
- Análisis contextual de resultados
- Explicaciones en lenguaje natural
- Recomendaciones personalizadas

## Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Para preguntas o soporte, por favor contactar a [tu-email@ejemplo.com](mailto:tu-email@ejemplo.com) 