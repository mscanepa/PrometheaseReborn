#!/usr/bin/env python3
"""
Polygenic Risk Score (PRS) Calculator

This script calculates a simplified polygenic risk score based on processed GWAS Catalog data
and user DNA data in MyHeritage/23andMe format. The PRS is calculated by combining effect
sizes from GWAS studies with individual genotype data.

Key features:
- Efficient data loading and merging
- Genotype encoding
- PRS calculation
- Error handling and logging
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(user_file: str, gwas_file: str) -> tuple:
    """
    Load and validate input data files.
    
    Args:
        user_file: Path to user DNA data file
        gwas_file: Path to processed GWAS Catalog file
        
    Returns:
        tuple: (user_data, gwas_data) DataFrames
    """
    try:
        logger.info("Loading input data...")
        
        # Verificar existencia de archivos
        if not os.path.exists(user_file):
            raise FileNotFoundError(f"User data file not found: {user_file}")
        if not os.path.exists(gwas_file):
            raise FileNotFoundError(f"GWAS data file not found: {gwas_file}")
        
        # Cargar datos con tipos explícitos
        user_data = pd.read_csv(
            user_file,
            sep='\t',
            low_memory=False,
            dtype={
                'rsid': str,
                'chromosome': str,
                'position': int,
                'genotype': str
            }
        )
        
        gwas_data = pd.read_csv(
            gwas_file,
            low_memory=False,
            dtype={
                'CHR_ID': str,
                'CHR_POS': int,
                'SNPS': str,
                'P-VALUE': float,
                'OR or BETA': str
            }
        )
        
        # Validar datos
        required_columns = ['rsid', 'chromosome', 'position', 'genotype']
        if not all(col in user_data.columns for col in required_columns):
            raise ValueError(f"User data missing required columns: {required_columns}")
        
        required_gwas_columns = ['CHR_ID', 'CHR_POS', 'SNPS', 'P-VALUE', 'OR or BETA']
        if not all(col in gwas_data.columns for col in required_gwas_columns):
            raise ValueError(f"GWAS data missing required columns: {required_gwas_columns}")
        
        logger.info("Successfully loaded and validated input data")
        return user_data, gwas_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def encode_genotype(genotype: str) -> int:
    """
    Convert genotype to numerical value.
    
    Current encoding scheme:
    - AA, CC, GG, TT = 0 (reference homozygote)
    - AG, CT, etc. = 1 (heterozygote)
    - GG, CC, AA, TT = 2 (alternate homozygote)
    
    Note: This encoding may need adjustment based on:
    1. The reference genome used
    2. The specific disease/trait being analyzed
    3. The effect direction in the GWAS study
    
    Args:
        genotype: String genotype (e.g., 'AA', 'AG', 'GG')
        
    Returns:
        int: Encoded genotype value
    """
    # Convert to uppercase for consistency
    genotype = genotype.upper()
    
    # Check for invalid genotypes
    if len(genotype) != 2 or not genotype.isalpha():
        return np.nan
    
    # Encode based on allele combinations
    if genotype[0] == genotype[1]:  # Homozygous
        return 0 if genotype in ['AA', 'CC', 'GG', 'TT'] else 2
    else:  # Heterozygous
        return 1

def calculate_prs(user_data: pd.DataFrame, gwas_data: pd.DataFrame) -> float:
    """
    Calculate polygenic risk score by combining GWAS effect sizes with genotype data.
    
    Args:
        user_data: DataFrame containing user genotype data
        gwas_data: DataFrame containing GWAS effect sizes
        
    Returns:
        float: Calculated PRS score
    """
    try:
        logger.info("Calculating PRS...")
        
        # Preparar datos
        user_data['chromosome'] = user_data['chromosome'].astype(str)
        gwas_data['CHR_ID'] = gwas_data['CHR_ID'].astype(str)
        
        # Filtrar SNPs significativos
        significant_snps = gwas_data[gwas_data['P-VALUE'] < 0.05]
        logger.info(f"Found {len(significant_snps)} significant SNPs")
        
        # Unir datos
        merged_data = pd.merge(
            user_data,
            significant_snps,
            left_on=['chromosome', 'position'],
            right_on=['CHR_ID', 'CHR_POS'],
            how='inner'
        )
        
        if len(merged_data) == 0:
            raise ValueError("No matching SNPs found between user data and GWAS data")
        
        logger.info(f"Successfully matched {len(merged_data)} SNPs")
        
        # Calcular PRS
        merged_data['effect_size'] = pd.to_numeric(merged_data['OR or BETA'], errors='coerce')
        merged_data['weighted_score'] = merged_data['effect_size'] * merged_data['P-VALUE']
        
        prs_score = merged_data['weighted_score'].sum()
        logger.info(f"Calculated PRS score: {prs_score}")
        
        return prs_score, merged_data
        
    except Exception as e:
        logger.error(f"Error calculating PRS: {str(e)}")
        raise

def main():
    """Función principal para calcular PRS."""
    try:
        # Verificar archivos de entrada
        user_file = 'data/raw/usuario_adaptado.txt'
        gwas_file = 'data/processed/gwas_catalog_processed.csv'
        
        if not os.path.exists(user_file):
            raise FileNotFoundError(f"User data file not found: {user_file}")
        if not os.path.exists(gwas_file):
            raise FileNotFoundError(f"GWAS data file not found: {gwas_file}")
        
        # Cargar datos
        user_data, gwas_data = load_data(user_file, gwas_file)
        
        # Calcular PRS
        prs_score, merged_data = calculate_prs(user_data, gwas_data)
        
        # Guardar resultados
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        merged_data.to_csv(os.path.join(output_dir, 'prs_results.csv'), index=False)
        
        with open(os.path.join(output_dir, 'prs_score.txt'), 'w') as f:
            f.write(f"PRS Score: {prs_score}\n")
            f.write(f"Number of SNPs analyzed: {len(merged_data)}\n")
        
        logger.info("PRS calculation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 