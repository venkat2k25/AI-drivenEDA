import streamlit as st
from utils.data_processor import DataProcessor
from utils.gemini_helper import GeminiAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re

def get_ai_visualization_suggestions(df, analyzer):
    """Get AI suggestions for visualizations based on data analysis"""
    prompt = f"""
    Analyze this dataset and suggest at least 6 insightful visualizations using ONLY these chart types:
    - Bar Chart (for comparisons between categories or numeric distributions)
    - Pie Chart (for showing proportions and percentages)
    - Histogram (for showing distribution of numeric data)

    Dataset Information:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Numeric Columns: {list(df.select_dtypes(include=['float64', 'int64']).columns)}
    - Categorical Columns: {list(df.select_dtypes(include=['object', 'category']).columns)}

    For each visualization, strictly use these tags and format:
    <CHART>
    TYPE: [chart type - must be Bar Chart, Pie Chart, or Histogram]
    COLUMNS: [column names separated by comma]
    TITLE: [chart title]
    PURPOSE: [purpose of visualization]
    </CHART>

    IMPORTANT:
    - Provide at least 6 visualizations
    - You can suggest multiple charts of the same type
    - For Bar Charts: suggest both single column and two-column comparisons
    - For Pie Charts: focus on categorical data with meaningful proportions
    - For Histograms: focus on numeric columns to show their distribution

    Make sure to include visualizations that show:
    1. Distribution of accident severity
    2. Vehicle types comparison
    3. Weather conditions analysis
    4. Time of day patterns
    5. Road type analysis
    6. Driver-related factors
    """
    
    try:
        suggestions = analyzer.suggest_visualization(df.columns, custom_prompt=prompt)
        return suggestions
    except Exception as e:
        return f"Error getting visualization suggestions: {str(e)}"

def parse_visualization_suggestions(suggestions_text):
    """Parse the AI suggestions using regex to extract visualization details"""

    chart_pattern = r'<CHART>(.*?)</CHART>'
    type_pattern = r'TYPE:\s*(.*?)(?:\n|$)'
    columns_pattern = r'COLUMNS:\s*(.*?)(?:\n|$)'
    title_pattern = r'TITLE:\s*(.*?)(?:\n|$)'
    purpose_pattern = r'PURPOSE:\s*(.*?)(?:\n|$)'
    
    charts = []

    chart_matches = re.finditer(chart_pattern, suggestions_text, re.DOTALL)
    
    for chart_match in chart_matches:
        chart_text = chart_match.group(1)

        chart_type = re.search(type_pattern, chart_text)
        columns = re.search(columns_pattern, chart_text)
        title = re.search(title_pattern, chart_text)
        purpose = re.search(purpose_pattern, chart_text)
        
        if all([chart_type, columns, title, purpose]):
            charts.append({
                'type': chart_type.group(1).strip(),
                'columns': [col.strip() for col in columns.group(1).split(',')],
                'title': title.group(1).strip(),
                'purpose': purpose.group(1).strip()
            })
    
    return charts

def create_visualization_from_suggestion(df, chart_info):
    """Create a visualization based on parsed suggestion"""
    try:
        chart_type = chart_info['type'].lower()
        columns = chart_info['columns']
        title = chart_info['title']
        
        if chart_type == "bar chart":
            if len(columns) == 2:

                fig = px.bar(df, x=columns[0], y=columns[1],
                           title=title, height=500,
                           color=columns[0]) 
            else:

                value_counts = df[columns[0]].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=title, height=500,
                           labels={'x': columns[0], 'y': 'Count'})
            
        elif chart_type == "pie chart":
            value_counts = df[columns[0]].value_counts()
            fig = px.pie(values=value_counts.values,
                        names=value_counts.index,
                        title=title, height=500)

            fig.update_traces(textposition='inside', textinfo='percent+label')
            
        elif chart_type == "histogram":
            fig = px.histogram(df, x=columns[0],
                             title=title, height=500,
                             nbins=30) 

            mean_val = df[columns[0]].mean()
            fig.add_vline(x=mean_val, line_dash="dash", 
                         annotation_text=f"Mean: {mean_val:.2f}")

        fig.update_layout(
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
            plot_bgcolor='rgba(0,0,0,0.02)',
            title_x=0.5, 
            title_font_size=16
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig, chart_info['purpose']
    except Exception as e:
        st.error(f"Error creating {chart_type}: {str(e)}")
        return None, None

def main():
    st.title("AI-Powered Data Visualization")

    if not st.session_state.get('data_cleaned', False):
        st.error(" Please clean your data first in the Data Cleaning page!")
        st.stop()
        return
    
    if 'cleaned_data' not in st.session_state:
        st.error(" No cleaned data found. Please process your data in the Data Cleaning page!")
        st.stop()
        return

    if 'visualization_generated' not in st.session_state:
        st.session_state.visualization_generated = False
        st.session_state.charts = None
        st.session_state.suggestions = None
    
    df = st.session_state['cleaned_data']

    st.subheader("Data Preview")
    st.dataframe(df.head())

    gemini_analyzer = GeminiAnalyzer(df)

    st.subheader("AI-Generated Visualizations")
    
    if not st.session_state.visualization_generated:
        with st.spinner("Generating visualizations..."):

            suggestions = get_ai_visualization_suggestions(df, gemini_analyzer)
            
            if isinstance(suggestions, str) and "Error" in suggestions:
                st.error(suggestions)
            else:

                charts = parse_visualization_suggestions(suggestions)
                
                if not charts:
                    st.warning("No valid visualization suggestions were found.")
                    return

                st.session_state.charts = charts
                st.session_state.suggestions = suggestions
                st.session_state.visualization_generated = True
                st.rerun()

    if st.session_state.visualization_generated and st.session_state.charts:
        st.write(f"Generated {len(st.session_state.charts)} visualizations based on AI analysis")

        if st.button("Regenerate Visualizations"):
            st.session_state.visualization_generated = False
            st.rerun()
            return

        for i, chart_info in enumerate(st.session_state.charts, 1):
            with st.expander(f" Visualization {i}: {chart_info['title']}", expanded=True):
                st.write(f"**Purpose:** {chart_info['purpose']}")
                fig, purpose = create_visualization_from_suggestion(df, chart_info)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        
        with st.expander("View AI Analysis Details"):
            st.text(st.session_state.suggestions)

if __name__ == "__main__":
    main() 