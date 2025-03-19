import pandas as pd
from fpdf import FPDF
import plotly.io as pio
import plotly.express as px
import google.generativeai as genai
import os
import tempfile

# Configure Gemini API
genai.configure(api_key="AIzaSyB3UUZ5sJCJTdNkJVF8RBICM7rw2_fCZQw")

class ReportGenerator:
    def __init__(self, original_df, cleaned_df):
        self.original_df = original_df
        self.cleaned_df = cleaned_df

    def _generate_summary_with_gemini(self):
        """Generate a comprehensive summary using Gemini API"""
        prompt = f"""
        Analyze the following dataset and provide a detailed summary with key insights:
        
        Original Dataset Shape: {self.original_df.shape}
        Cleaned Dataset Shape: {self.cleaned_df.shape}
        
        Column Information:
        {str(self.cleaned_df.dtypes)}
        
        Provide:
        1. Dataset Overview with key insights
        2. summary about data Like what data is used
        3. Statistical patterns for the first 2 numeric columns
        4. trend on data
        """

  
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    def generate_report(self):
        """Generate a PDF report with dataset analysis using Gemini API"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'AI-Powered Dataset Analysis Report', ln=True, align='C')
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '1. Dataset Overview', ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Original Shape: {self.original_df.shape}', ln=True)
        pdf.cell(0, 10, f'Cleaned Shape: {self.cleaned_df.shape}', ln=True)
        pdf.ln(5)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '2. Data Quality Analysis', ln=True)
        pdf.set_font('Arial', '', 12)
        
        missing_original = self.original_df.isnull().sum().sum()
        missing_cleaned = self.cleaned_df.isnull().sum().sum()
        pdf.cell(0, 10, f'Missing Values (Original): {missing_original}', ln=True)
        pdf.cell(0, 10, f'Missing Values (Cleaned): {missing_cleaned}', ln=True)
        pdf.ln(5)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '3. Statistical Summary', ln=True)
        pdf.set_font('Arial', '', 12)

        numeric_cols = self.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            stats = self.cleaned_df[numeric_cols].describe()
            for col in numeric_cols[:2]:  # First 2 numeric columns only
                pdf.cell(0, 10, f'\nStatistics for {col}:', ln=True)
                for stat, value in stats[col].items():
                    pdf.cell(0, 10, f'- {stat}: {value:.2f}', ln=True)
                pdf.ln(5)

        # 4. Visualizations
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '4. Key Visualizations', ln=True)

        # Generate and embed visualizations
        with tempfile.TemporaryDirectory() as tmpdirname:
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                fig = px.histogram(self.cleaned_df, x=col, title=f'Distribution of {col}')
                temp_path = os.path.join(tmpdirname, f'{col}_dist.png')
                pio.write_image(fig, temp_path)
                pdf.image(temp_path, x=10, w=190)
                pdf.ln(5)

        # 5. Recommendations
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '5. AI-Generated Recommendations', ln=True)
        pdf.set_font('Arial', '', 12)

        # Generate AI-based insights using Gemini API
        ai_insights = self._generate_summary_with_gemini()
        pdf.multi_cell(0, 10, ai_insights)

        # 6. Executive Summary
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '6. Executive Summary', ln=True)
        pdf.set_font('Arial', '', 12)

        summary_points = [
            "Implement feature engineering for meaningful variables.",
            "Set up automated data validation pipelines.",
            "Regularly monitor data quality with visual dashboards.",
            "Consider adding domain-specific feature enhancements.",
            "Create alerts for anomaly detection in real-time."
        ]
        
        for point in summary_points:
            pdf.cell(0, 10, f'- {point}', ln=True)

        try:
            return pdf.output(dest='S').encode('latin1')
        except UnicodeEncodeError:
            return pdf.output(dest='S').encode('latin1', errors='replace')
