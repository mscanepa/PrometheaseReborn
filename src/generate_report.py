#!/usr/bin/env python3
"""
Report Generation Module for TheModernPromethease

This module generates comprehensive reports from PRS calculations and processed data.
It creates detailed reports with visualizations and interpretations of the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class to handle report generation and visualization."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the report generator.
        
        Args:
            data_dir: Directory containing processed data and PRS results
        """
        self.data_dir = Path(data_dir)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load necessary data
        self.prs_results = None
        self.gwas_data = None
        self.clinvar_data = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load all necessary data files."""
        try:
            # Load PRS results
            prs_path = self.data_dir / "prs_results.csv"
            if prs_path.exists():
                self.prs_results = pd.read_csv(prs_path)
                logger.info("Successfully loaded PRS results")
            
            # Load GWAS data
            gwas_path = self.data_dir / "gwas_processed.csv"
            if gwas_path.exists():
                self.gwas_data = pd.read_csv(gwas_path)
                logger.info("Successfully loaded GWAS data")
            
            # Load ClinVar data
            clinvar_path = self.data_dir / "clinvar_processed.csv"
            if clinvar_path.exists():
                self.clinvar_data = pd.read_csv(clinvar_path)
                logger.info("Successfully loaded ClinVar data")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_prs_distribution_plot(self, output_path: Path) -> None:
        """
        Generate a distribution plot of PRS scores.
        
        Args:
            output_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.prs_results, x='prs', bins=30)
            plt.title('Distribution of Polygenic Risk Scores')
            plt.xlabel('PRS Score')
            plt.ylabel('Count')
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Generated PRS distribution plot: {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating PRS distribution plot: {str(e)}")
    
    def generate_trait_risk_plot(self, output_path: Path) -> None:
        """
        Generate a bar plot of top risk traits.
        
        Args:
            output_path: Path to save the plot
        """
        try:
            # Get top 10 traits by PRS
            top_traits = self.prs_results.nlargest(10, 'prs')
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_traits, x='trait', y='prs')
            plt.title('Top 10 Traits by Polygenic Risk Score')
            plt.xlabel('Trait')
            plt.ylabel('PRS Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Generated trait risk plot: {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating trait risk plot: {str(e)}")
    
    def generate_html_report(self, output_path: Path) -> None:
        """
        Generate a comprehensive HTML report.
        
        Args:
            output_path: Path to save the HTML report
        """
        try:
            # Create report content
            report_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>TheModernPromethease Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .section {{ margin-bottom: 30px; }}
                    .plot {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>TheModernPromethease Genomic Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <p>Number of traits analyzed: {len(self.prs_results)}</p>
                    <p>Average PRS: {self.prs_results['prs'].mean():.4f}</p>
                    <p>Standard deviation: {self.prs_results['prs'].std():.4f}</p>
                </div>
                
                <div class="section">
                    <h2>Top Risk Traits</h2>
                    <table>
                        <tr>
                            <th>Trait</th>
                            <th>PRS Score</th>
                            <th>Number of SNPs</th>
                        </tr>
            """
            
            # Add top traits to table
            top_traits = self.prs_results.nlargest(10, 'prs')
            for _, row in top_traits.iterrows():
                report_content += f"""
                    <tr>
                        <td>{row['trait']}</td>
                        <td>{row['prs']:.4f}</td>
                        <td>{row['num_snps']}</td>
                    </tr>
                """
            
            # Add plots to report
            report_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>PRS Distribution</h2>
                    <div class="plot">
                        <img src="prs_distribution.png" alt="PRS Distribution" style="width:100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Top Risk Traits Visualization</h2>
                    <div class="plot">
                        <img src="trait_risk.png" alt="Top Risk Traits" style="width:100%;">
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML report
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Generated HTML report: {output_path}")
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
    
    def generate_all_reports(self) -> None:
        """Generate all reports and visualizations."""
        try:
            # Create timestamp for report directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = self.reports_dir / f"report_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            self.generate_prs_distribution_plot(report_dir / "prs_distribution.png")
            self.generate_trait_risk_plot(report_dir / "trait_risk.png")
            
            # Generate HTML report
            self.generate_html_report(report_dir / "report.html")
            
            logger.info(f"All reports generated in: {report_dir}")
            
            return report_dir
        
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return None

def main():
    """Main function to execute report generation."""
    try:
        # Initialize report generator
        generator = ReportGenerator()
        
        # Generate all reports
        report_dir = generator.generate_all_reports()
        
        if report_dir:
            print(f"\nReport Generation Summary:")
            print("------------------------")
            print(f"Reports generated in: {report_dir}")
            print(f"Number of traits analyzed: {len(generator.prs_results)}")
            print(f"Report files generated:")
            for file in report_dir.glob("*"):
                print(f"  - {file.name}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 