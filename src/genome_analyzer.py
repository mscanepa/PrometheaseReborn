#!/usr/bin/env python3
"""
Genome Analyzer

This module handles the analysis of genome data and generates reports
integrating information from multiple clinical databases.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sqlalchemy import create_engine, text
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('genome_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenomeAnalyzer:
    def __init__(self, db_url: str):
        """Initialize the genome analyzer with database connection."""
        self.engine = create_engine(db_url)
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
    
    def process_genome_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process a genome file and extract relevant SNPs."""
        try:
            logger.info(f"Processing genome file: {file_path}")
            
            # Read the genome file
            df = pd.read_csv(file_path, sep='\t', comment='#', 
                           names=['rsid', 'chromosome', 'position', 'genotype'])
            
            # Basic cleaning
            df = df.dropna()
            df['rsid'] = df['rsid'].str.strip()
            df['genotype'] = df['genotype'].str.strip()
            
            # Filter out invalid genotypes
            valid_genotypes = df['genotype'].str.match(r'^[ATCG][ATCG]$')
            df = df[valid_genotypes]
            
            logger.info(f"Processed {len(df)} SNPs from genome file")
            return df
            
        except Exception as e:
            logger.error(f"Error processing genome file: {str(e)}")
            return None
    
    def query_clinical_data(self, rsids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Query clinical databases for the given RSIDs."""
        try:
            with self.engine.connect() as conn:
                # Query GWAS data
                gwas_query = text("""
                    SELECT 
                        snp_id,
                        disease_trait,
                        p_value,
                        or_beta
                    FROM gwas_variants
                    WHERE snp_id = ANY(:rsids)
                """)
                gwas_data = pd.read_sql(gwas_query, conn, params={'rsids': rsids})
                
                # Query ClinVar data
                clinvar_query = text("""
                    SELECT 
                        name,
                        clinical_significance,
                        phenotype,
                        chromosome,
                        position,
                        reference_allele,
                        alternate_allele
                    FROM clinvar_variants
                    WHERE name = ANY(:rsids)
                """)
                clinvar_data = pd.read_sql(clinvar_query, conn, params={'rsids': rsids})
                
                # Query dbSNP data
                dbsnp_query = text("""
                    SELECT 
                        rs_id,
                        chromosome,
                        position,
                        reference_allele,
                        alternate_allele,
                        allele_frequency,
                        population
                    FROM dbsnp_variants
                    WHERE rs_id = ANY(:rsids)
                """)
                dbsnp_data = pd.read_sql(dbsnp_query, conn, params={'rsids': rsids})
                
                return gwas_data, clinvar_data, dbsnp_data
                
        except Exception as e:
            logger.error(f"Error querying clinical data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def generate_report(self, genome_data: pd.DataFrame, 
                       gwas_data: pd.DataFrame,
                       clinvar_data: pd.DataFrame,
                       dbsnp_data: pd.DataFrame) -> str:
        """Generate an HTML report with the analysis results."""
        try:
            # Create a timestamp for the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.report_dir / f"genome_report_{timestamp}.html"
            
            # Merge all data
            merged_data = genome_data.copy()
            
            # Merge with GWAS data
            if not gwas_data.empty:
                merged_data = merged_data.merge(
                    gwas_data,
                    left_on='rsid',
                    right_on='snp_id',
                    how='left'
                )
            
            # Merge with ClinVar data
            if not clinvar_data.empty:
                merged_data = merged_data.merge(
                    clinvar_data,
                    left_on='rsid',
                    right_on='name',
                    how='left'
                )
            
            # Merge with dbSNP data
            if not dbsnp_data.empty:
                merged_data = merged_data.merge(
                    dbsnp_data,
                    left_on='rsid',
                    right_on='rs_id',
                    how='left'
                )
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Genome Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .highlight {{ background-color: #fff3cd; }}
                </style>
            </head>
            <body>
                <h1>Genome Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total SNPs analyzed: {len(genome_data)}</p>
                
                <table>
                    <tr>
                        <th>RSID</th>
                        <th>Chromosome</th>
                        <th>Position</th>
                        <th>Genotype</th>
                        <th>GWAS Traits</th>
                        <th>Clinical Significance</th>
                        <th>Phenotype</th>
                        <th>Allele Frequency</th>
                        <th>Population</th>
                    </tr>
            """
            
            # Add rows to the table
            for _, row in merged_data.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['rsid']}</td>
                        <td>{row['chromosome']}</td>
                        <td>{row['position']}</td>
                        <td>{row['genotype']}</td>
                        <td>{row.get('disease_trait', '')}</td>
                        <td>{row.get('clinical_significance', '')}</td>
                        <td>{row.get('phenotype', '')}</td>
                        <td>{row.get('allele_frequency', '')}</td>
                        <td>{row.get('population', '')}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            # Save the report
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Report generated successfully: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return ""
    
    def analyze_genome(self, file_path: str) -> Optional[str]:
        """Main function to analyze a genome file and generate a report."""
        try:
            # Step 1: Process the genome file
            genome_data = self.process_genome_file(file_path)
            if genome_data is None or genome_data.empty:
                logger.error("No valid genome data found")
                return None
            
            # Step 2: Get RSIDs to query
            rsids = genome_data['rsid'].tolist()
            
            # Step 3: Query clinical databases
            gwas_data, clinvar_data, dbsnp_data = self.query_clinical_data(rsids)
            
            # Step 4: Generate report
            report_path = self.generate_report(genome_data, gwas_data, clinvar_data, dbsnp_data)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error analyzing genome: {str(e)}")
            return None

def main():
    """Main function to run genome analysis."""
    if len(sys.argv) != 2:
        print("Usage: python genome_analyzer.py <genome_file_path>")
        sys.exit(1)
    
    genome_file = sys.argv[1]
    if not os.path.exists(genome_file):
        print(f"Error: File {genome_file} not found")
        sys.exit(1)
    
    # Load database URL from environment
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/genealogy')
    
    # Create analyzer and process genome
    analyzer = GenomeAnalyzer(db_url)
    report_path = analyzer.analyze_genome(genome_file)
    
    if report_path:
        print(f"Analysis complete. Report generated at: {report_path}")
    else:
        print("Error generating report")

if __name__ == "__main__":
    main() 