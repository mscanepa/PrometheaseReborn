#!/usr/bin/env python3
"""
Database Optimizer

Este módulo se encarga de optimizar las consultas de la base de datos mediante
la creación y gestión de índices optimizados. No modifica los datos existentes,
solo mejora el rendimiento de las consultas.
"""

import logging
from sqlalchemy import create_engine, text
from typing import List, Dict
import os
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv(dotenv_path='../../relationship-calculator-api/.env')

class DatabaseOptimizer:
    def __init__(self, db_url: str = None):
        """Inicializa el optimizador de base de datos."""
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/genealogy')
        self.engine = create_engine(self.db_url)
        
    def create_indexes(self) -> bool:
        """Crea los índices optimizados para las tablas principales."""
        try:
            with self.engine.connect() as conn:
                # Índices para gwas_variants
                conn.execute(text("""
                    -- Optimiza búsquedas por cromosoma y posición
                    CREATE INDEX IF NOT EXISTS idx_gwas_chr_pos ON gwas_variants (chromosome, position);
                    
                    -- Optimiza búsquedas rápidas por SNP ID
                    CREATE INDEX IF NOT EXISTS idx_gwas_snp_id ON gwas_variants (snp_id);
                """))
                
                # Índices para dbsnp_variants
                conn.execute(text("""
                    -- Búsqueda eficiente por cromosoma y posición
                    CREATE INDEX IF NOT EXISTS idx_dbsnp_chr_pos ON dbsnp_variants (chromosome, position);
                    
                    -- Búsqueda rápida por rsID
                    CREATE INDEX IF NOT EXISTS idx_dbsnp_rs_id ON dbsnp_variants (rs_id);
                    
                    -- Mejora consultas frecuentes para población específica
                    CREATE INDEX IF NOT EXISTS idx_dbsnp_population ON dbsnp_variants (population);
                """))
                
                # Índices para clinvar_variants
                conn.execute(text("""
                    -- Consultas rápidas por cromosoma y posición
                    CREATE INDEX IF NOT EXISTS idx_clinvar_chr_pos ON clinvar_variants (chromosome, position);
                    
                    -- Búsqueda eficiente por Variation ID
                    CREATE INDEX IF NOT EXISTS idx_clinvar_variation_id ON clinvar_variants (variation_id);
                    
                    -- Consultas frecuentes por relevancia clínica
                    CREATE INDEX IF NOT EXISTS idx_clinvar_clinical_significance ON clinvar_variants (clinical_significance);
                """))
                
                conn.commit()
                logger.info("Índices creados exitosamente")
                return True
                
        except Exception as e:
            logger.error(f"Error al crear índices: {str(e)}")
            return False
    
    def verify_indexes(self) -> Dict[str, List[Dict]]:
        """Verifica los índices existentes en las tablas principales."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE tablename IN ('gwas_variants', 'dbsnp_variants', 'clinvar_variants')
                    ORDER BY tablename, indexname;
                """))
                
                indexes = {}
                for row in result:
                    table = row.tablename
                    if table not in indexes:
                        indexes[table] = []
                    indexes[table].append({
                        'name': row.indexname,
                        'definition': row.indexdef
                    })
                
                return indexes
                
        except Exception as e:
            logger.error(f"Error al verificar índices: {str(e)}")
            return {}
    
    def analyze_table_performance(self, table_name: str) -> Dict:
        """Analiza el rendimiento de una tabla específica."""
        try:
            with self.engine.connect() as conn:
                # Obtener estadísticas de la tabla
                stats = conn.execute(text(f"""
                    SELECT 
                        relname as table_name,
                        n_live_tup as row_count,
                        pg_size_pretty(pg_total_relation_size(relid)) as total_size,
                        pg_size_pretty(pg_indexes_size(relid)) as index_size
                    FROM pg_stat_user_tables
                    WHERE relname = :table_name;
                """), {"table_name": table_name}).fetchone()
                
                # Obtener información de índices
                indexes = conn.execute(text(f"""
                    SELECT 
                        indexname,
                        indexdef,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_indexes
                    WHERE tablename = :table_name;
                """), {"table_name": table_name}).fetchall()
                
                return {
                    'table_stats': {
                        'row_count': stats.row_count,
                        'total_size': stats.total_size,
                        'index_size': stats.index_size
                    },
                    'indexes': [
                        {
                            'name': idx.indexname,
                            'definition': idx.indexdef,
                            'size': idx.index_size
                        } for idx in indexes
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error al analizar rendimiento de {table_name}: {str(e)}")
            return {}

def main():
    """Función principal para ejecutar la optimización."""
    optimizer = DatabaseOptimizer()
    
    # Verificar índices existentes
    logger.info("Verificando índices existentes...")
    current_indexes = optimizer.verify_indexes()
    for table, indexes in current_indexes.items():
        logger.info(f"\nÍndices en {table}:")
        for idx in indexes:
            logger.info(f"- {idx['name']}: {idx['definition']}")
    
    # Crear nuevos índices
    logger.info("\nCreando nuevos índices...")
    if optimizer.create_indexes():
        logger.info("Índices creados exitosamente")
    else:
        logger.error("Error al crear índices")
    
    # Analizar rendimiento
    logger.info("\nAnalizando rendimiento de tablas...")
    for table in ['gwas_variants', 'dbsnp_variants', 'clinvar_variants']:
        stats = optimizer.analyze_table_performance(table)
        logger.info(f"\nEstadísticas de {table}:")
        logger.info(f"- Filas: {stats['table_stats']['row_count']}")
        logger.info(f"- Tamaño total: {stats['table_stats']['total_size']}")
        logger.info(f"- Tamaño de índices: {stats['table_stats']['index_size']}")
        logger.info("Índices:")
        for idx in stats['indexes']:
            logger.info(f"- {idx['name']}: {idx['size']}")

if __name__ == "__main__":
    main() 