import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.gemini_helper import GeminiAnalyzer
from utils.report_generator import ReportGenerator
import base64

def get_ai_insights(df):
    """Get AI insights about data quality issues"""
    analyzer = GeminiAnalyzer(df)
    prompt = """Analyze this dataset for data quality issues. Consider:
    1. Missing values and their patterns
    2. Potential outliers
    3. Data type inconsistencies
    4. Value distributions
    Provide specific recommendations for cleaning this dataset."""
    
    try:
        insights = analyzer.suggest_visualization(df.columns)
        return insights
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

def clean_dataset(df):
    """Clean the dataset using specified methods"""
    cleaned_df = df.copy()
    original_index = cleaned_df.index.copy()  

    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns

    if st.checkbox("Handle Missing Values"):

        if len(categorical_cols) > 0:
            for col in categorical_cols:
                missing_mask = cleaned_df[col].isnull()
                if missing_mask.any():
                    most_frequent = cleaned_df[col].mode()[0]
                    cleaned_df.loc[missing_mask, col] = most_frequent
                    st.info(f"Filled {missing_mask.sum()} missing values in {col} with '{most_frequent}'")
            st.success("Filled missing categorical values with most frequent values")

        if len(numeric_cols) > 0:
            n_neighbors = st.slider("Select number of neighbors for KNN", 1, 10, 5)
            
            for col in numeric_cols:
                missing_mask = cleaned_df[col].isnull()
                if missing_mask.any():

                    non_missing_values = cleaned_df.loc[~missing_mask, col].copy()

                    temp_df = cleaned_df[numeric_cols].copy()

                    imputer = KNNImputer(n_neighbors=min(n_neighbors, (~missing_mask).sum()))
                    imputed_values = imputer.fit_transform(temp_df)

                    # If all values in the original column are integers, round the imputed values
                    if pd.api.types.is_integer_dtype(df[col]):
                        cleaned_df.loc[missing_mask, col] = np.round(imputed_values[missing_mask, list(numeric_cols).index(col)])
                    else:
                        cleaned_df.loc[missing_mask, col] = imputed_values[missing_mask, list(numeric_cols).index(col)]

                    cleaned_df.loc[~missing_mask, col] = non_missing_values

                    if non_missing_values.min() >= 0:
                        negative_mask = (cleaned_df[col] < 0) & missing_mask
                        if negative_mask.any():
                            col_mean = non_missing_values.mean()
                            cleaned_df.loc[negative_mask, col] = np.round(col_mean) if pd.api.types.is_integer_dtype(df[col]) else col_mean
                            st.warning(f"Fixed negative imputed values in {col} by replacing with mean ({col_mean:.2f})")
                    
                    st.info(f"Filled {missing_mask.sum()} missing values in {col}")
            
            st.success("Completed numeric value imputation")

    if st.checkbox("Remove Outliers (Isolation Forest)"):
        if len(numeric_cols) > 0:
            contamination = st.slider("Select contamination factor", 0.01, 0.2, 0.1, 0.01,
                                   help="The proportion of outliers in the data (between 0.01 and 0.2)")
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(cleaned_df[numeric_cols])
            n_outliers = (outlier_labels == -1).sum()
            
            st.write(f"Found {n_outliers} potential outliers ({(n_outliers/len(cleaned_df)*100):.1f}% of data)")
            
            if st.button("Remove Detected Outliers"):
                cleaned_df = cleaned_df[outlier_labels == 1]
                st.success(f"Removed {n_outliers} outliers. Rows remaining: {len(cleaned_df)}")

    st.subheader("Missing Values Summary")
    missing_before = df.isnull().sum()
    missing_after = cleaned_df.isnull().sum()
    
    if missing_before.any():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Before Cleaning:")
            st.write(missing_before[missing_before > 0])
        with col2:
            st.write("After Cleaning:")
            st.write(missing_after[missing_before > 0])
        with col3:
            st.write("Values Filled:")
            filled = missing_before - missing_after
            st.write(filled[filled > 0])

        common_index = cleaned_df.index.intersection(original_index)
        
        for col in df.columns:
            non_missing_mask = ~df.loc[common_index, col].isnull()
            if non_missing_mask.any():
                original_values = df.loc[common_index, col][non_missing_mask]
                cleaned_values = cleaned_df.loc[common_index, col][non_missing_mask]
                
                try:
                    if not np.array_equal(original_values.values, cleaned_values.values):
                        st.error(f"Warning: Some non-missing values were changed in {col}!")
                        value_diff = (original_values != cleaned_values)
                        if value_diff.any():
                            st.write("Sample of changed values:")
                            comparison = pd.DataFrame({
                                'Original': original_values[value_diff],
                                'Cleaned': cleaned_values[value_diff]
                            }).head()
                            st.write(comparison)
                except Exception as e:
                    st.error(f"Error comparing values in {col}: {str(e)}")
    else:
        st.write("No missing values found in the dataset")
    
    return cleaned_df


def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download Report</a>'

def main():
    st.title(" Data Cleaning")
    

    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    

    if st.session_state.cleaned_data is not None:
        st.sidebar.markdown("### Download Data")
        csv = st.session_state.cleaned_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            help="Download the current cleaned dataset as CSV",
            key="sidebar_download"
        )

    st.subheader("1. Load Dataset")

    if not st.session_state.data_cleaned:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_data = df 
            
            st.write("Dataset Preview:", df.head())
            st.write("Shape:", df.shape)
            
             
            st.subheader("2. Get AI Insights")
            if st.button("Analyze Data Quality"):
                with st.spinner("Getting AI insights..."):
                    insights = get_ai_insights(df)
                    st.write(insights)
            
           
            st.subheader("3. Clean Data")
            st.write("Select cleaning operations:")
            
            cleaned_df = clean_dataset(df)
            
            
            if st.checkbox("Show Cleaning Results"):
                st.subheader("4. Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Original Data Info:")
                    st.write(f"- Rows: {df.shape[0]}")
                    st.write(f"- Missing Values: {df.isnull().sum().sum()}")
                
                with col2:
                    st.write("Cleaned Data Info:")
                    st.write(f"- Rows: {cleaned_df.shape[0]}")
                    st.write(f"- Missing Values: {cleaned_df.isnull().sum().sum()}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm and Save Cleaned Data"):
                    st.session_state.cleaned_data = cleaned_df
                    st.session_state.data_cleaned = True
                    st.success("✅ Data cleaned and saved! You can now proceed to AI Analysis or Visualization pages.")
                    st.rerun()
            
            with col2:

                if 'cleaned_df' in locals():
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="Download Current Cleaned Data",
                        data=csv,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                        help="Download the cleaned dataset without saving",
                        key="temp_download"
                    )
    
    else:

        st.success("Data has been cleaned and is ready for analysis!")
        

        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.cleaned_data.head())
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        
        col1, col2 = st.columns(2)
        with col1:

            if st.button("Reset and Clean New Data"):
                st.session_state.data_cleaned = False
                st.session_state.original_data = None
                st.session_state.cleaned_data = None
                st.rerun()
        
        with col2:

            csv = st.session_state.cleaned_data.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                help="Download the current cleaned dataset as CSV",
                key="main_download"
            )

    if st.session_state.data_cleaned:
        st.subheader("6. Generate Report")
        if st.button("Generate Analysis Report"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    report_gen = ReportGenerator(
                        st.session_state.original_data,
                        st.session_state.cleaned_data
                    )
                    pdf = report_gen.generate_report()
                    html = create_download_link(pdf, "dataset_analysis_report.pdf")
                    st.markdown(html, unsafe_allow_html=True)
                    st.success("Report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main() 