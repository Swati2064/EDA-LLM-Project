import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

# Function to Perform EDA and Generate Visualizations
def eda_analysis(file_path):

    # Load Dataset
    df = pd.read_csv(file_path)

    # Fill missing values for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fill missing values for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Dataset Summary
    summary = df.describe(include='all').to_string()

    # Missing Values
    missing_values = df.isnull().sum().to_string()

    # Generate AI Insights
    insights = generate_ai_insights(summary)

    # Generate Visualizations
    plot_paths = generate_visualizations(df)

    # Final Report
    report = f"""
✅ Data Loaded Successfully!

📌 Dataset Summary:
{summary}

📌 Missing Values:
{missing_values}

📌 AI Insights:
{insights}
"""

    return report, plot_paths


# AI Insights using Ollama Gemma Model
def generate_ai_insights(df_summary):

    prompt = f"""
    Analyze the following dataset summary and provide key insights:

    {df_summary}
    """

    response = ollama.chat(
        model="gemma3:270m",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response['message']['content']


# Function to Generate Visualizations
def generate_visualizations(df):

    plot_paths = []

    # Histograms for Numeric Columns
    for col in df.select_dtypes(include=['number']).columns:

        plt.figure(figsize=(6, 4))

        sns.histplot(df[col], bins=30, kde=True)

        plt.title(f"Distribution of {col}")

        path = f"{col}_distribution.png"

        plt.savefig(path)

        plot_paths.append(path)

        plt.close()

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=['number'])

    if not numeric_df.empty:

        plt.figure(figsize=(10, 6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")

        path = "correlation_heatmap.png"

        plt.savefig(path)

        plot_paths.append(path)

        plt.close()

    return plot_paths


# Gradio Interface
demo = gr.Interface(
    fn=eda_analysis,

    inputs=gr.File(
        type="filepath",
        label="Upload CSV File"
    ),

    outputs=[
        gr.Textbox(label="EDA Report"),
        gr.Gallery(label="Data Visualizations")
    ],

    title="📊 LLM-Powered Exploratory Data Analysis (EDA)",

    description="""
    Upload a CSV dataset and get:
    - Automated EDA
    - Missing value analysis
    - AI-generated insights
    - Data visualizations
    """
)

# Launch App
demo.launch(share=True)