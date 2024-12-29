import streamlit as st
import google.generativeai as genai
import os
import docx2txt
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import json
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('Google_api_key'))

# Gemini AI Configuration
generation_config = {
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2000,
}



# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = pdf.PdfReader(uploaded_file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += str(page.extract_text())
        return text_content
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to extract text from DOCX
def extract_text_from_docx(upload_file):
    try:
        text = docx2txt.process(upload_file)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

# Function to create word cloud
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Function to calculate readability score
def calculate_readability_score(text):
    words = len(re.findall(r'\w+', text))
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences == 0:
        sentences = 1
    avg_words_per_sentence = words / sentences
    return min(100, max(0, 100 - (abs(avg_words_per_sentence - 20) * 2)))

# Function to create radar chart
def create_radar_chart(skills_data):
    categories = list(skills_data.keys())
    values = list(skills_data.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False
    )
    return fig

# Function to create timeline chart
import pandas as pd
from datetime import datetime


def create_timeline_chart(experience_data):
    # Create a copy of the dataframe to avoid modifying the original
    df = experience_data.copy()

    # Function to parse date strings
    def parse_date(date_str):
        if pd.isna(date_str) or date_str.lower() == 'present':
            return datetime.now()
        try:
            return pd.to_datetime(date_str, format='%B %Y')
        except:
            try:
                return pd.to_datetime(date_str)
            except:
                return None

    # Convert date columns
    df['start_date'] = df['start_date'].apply(parse_date)
    df['end_date'] = df['end_date'].apply(parse_date)

    # Create the timeline
    fig = px.timeline(df,
                      x_start='start_date',
                      x_end='end_date',
                      y='role',
                      title='Experience Timeline')

    # Customize the layout
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis=dict(
            title="Date",
            type="date"
        )
    )

    return fig
# Prompt template
input_prompt_template = """
Analyze this resume against the job description. Focus especially on certifications, industry gaps, and development areas.

Resume: {text}
Job Description: {job_description}
Job Title: {job_title}

Provide specific recommendations in this exact JSON format:
{{
    "match_percentage": "80",
    "technical_skills": {{
        "matching_skills": ["skill1", "skill2"],
        "missing_skills": ["skill3", "skill4"],
        "skill_match_percentage": "75"
    }},
    "experience_analysis": {{
        "years_of_experience": "5",
        "experience_match_percentage": "70",
        "key_achievements": ["achievement1", "achievement2"],
        "career_progression": [
            {{
                "role": "Software Engineer",
                "start_date": "2020",
                "end_date": "2023"
            }}
        ]
    }},
    "education_analysis": {{
        "education_match": "85",
        "recommended_certifications": [
            "Specific certification related to job role",
            "Industry-standard certification in the field",
            "Technical certification for missing skills",
            "Professional certification for career growth"
        ]
    }},
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "ats_compatibility": {{
        "format_score": "90",
        "issues_found": ["issue1", "issue2"],
        "formatting_suggestions": ["format1", "format2"]
    }},
    "cultural_fit": {{
        "alignment_score": "80",
        "matching_values": ["value1", "value2"],
        "development_areas": [
            "Specific development area based on job requirements",
            "Soft skill that needs improvement",
            "Technical area that needs strengthening",
            "Professional development opportunity"
        ]
    }},
    "industry_analysis": {{
        "industry_alignment": "75",
        "relevant_experience": ["exp1", "exp2"],
        "industry_gaps": [
            "Specific industry knowledge gap",
            "Missing industry-specific experience",
            "Required industry certification or training",
            "Industry-specific tool or methodology"
        ]
    }}
}}

Important:
1. For recommended_certifications: Include specific, relevant certifications based on the job requirements and industry standards. If none, return an empty array.
2. For industry_gaps: List specific missing industry knowledge, experiences, or qualifications required for the role. If none, return an empty array.
3. For development_areas: Identify specific skills, competencies, or experiences that would enhance the candidate's fit for the role. If none, return an empty array.
4. Ensure all arrays contain meaningful, specific items related to the job and industry."""

# Function to generate response from Gemini
def generate_response(input_text):
    try:
        llm = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
        output = llm.generate_content(input_text)

        # Clean the response
        response_text = output.text.strip()
        response_text = response_text.replace('```json', '').replace('```', '')

        # Debug: Log the raw response
        #st.write("Raw Response from Gemini:")
        #st.code(response_text)

        # Validate JSON
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError as je:
            st.error(f"Invalid JSON format received. Please try again.")
            st.code(response_text)
            return None

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to get color mode
def get_color_mode():
    try:
        return st.session_state.color_mode
    except:
        st.session_state.color_mode = "light"
        return "light"

# Main Streamlit app


def main():
    st.set_page_config(page_title="Resume Analyzer Pro", layout="wide")

    # Modern CSS styling
    st.markdown("""
            <style>
            [data-theme="light"] {
                --bg-color: #ffffff;
                --text-color: #333333;
                --card-bg: #ffffff;
                --shadow: rgba(0,0,0,0.1);
            }
            [data-theme="dark"] {
                --bg-color: #1e1e1e;
                --text-color: #ffffff;
                --card-bg: #2d2d2d;
                --shadow: rgba(255,255,255,0.1);
            }
            .main-header {
                background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 15px var(--shadow);
            }
            .card {
                background: var(--card-bg);
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px var(--shadow);
                margin-bottom: 1rem;
                transition: transform 0.3s ease;
                color: var(--text-color);
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .stButton>button {
                background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                font-weight: 500;
                transition: all 0.3s ease;
                width: 100%;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px var(--shadow);
            }
            </style>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.session_state.color_mode = theme.lower()

        st.markdown("### 📤 Export Options")
        export_format = st.selectbox("Export Format", ["PDF", "DOCX", "JSON"])
        if st.button("Export Analysis"):
            if 'analysis' in locals():
                with st.spinner("Preparing export..."):
                    time.sleep(1)  # Simulate export
                    st.success("Analysis exported successfully!")
            else:
                st.warning("Please analyze a resume first")

    ## Header with Lottie Animation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div class="main-header">
                <h1>📊 Resume Analyzer Pro</h1>
                <p>AI-Powered Resume Analysis & Career Insights</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
        lottie_json = load_lottie_url(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=150)

    # Tabs
    tabs = st.tabs(["🎯 Analysis", "🔍 ATS Check", "📈 Insights"])

    with tabs[0]:
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
                job_description = st.text_area(
                    "Job Description",
                    height=200,
                    placeholder="Paste the complete job description here..."
                )
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload Your Resume",
                    type=["pdf", "docx"],
                    help="Supported formats: PDF, DOCX"
                )
                if uploaded_file:
                    with st.spinner("Processing file..."):
                        time.sleep(1)
                        st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Analyze Resume", type="primary"):
            if job_description and uploaded_file and job_title:
                with st.spinner("Analyzing your resume..."):
                    # Progress bar animation
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    # File processing
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = extract_text_from_docx(uploaded_file)
                    else:
                        st.error("Invalid File Format")
                        st.stop()

                    if text:
                        response = generate_response(
                            input_prompt_template.format(
                                text=text,
                                job_description=job_description,
                                job_title=job_title
                            )
                        )

                        if response:
                            try:
                                analysis = json.loads(response)

                                # Results display
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Match", f"{analysis['match_percentage']}%")
                                with col2:
                                    st.metric("Skills", f"{analysis['technical_skills']['skill_match_percentage']}%")
                                with col3:
                                    st.metric("Industry", f"{analysis['industry_analysis']['industry_alignment']}%")
                                with col4:
                                    st.metric("Culture", f"{analysis['cultural_fit']['alignment_score']}%")
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Skills Analysis
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.subheader("💡 Skills Analysis")
                                skills_data = {
                                    "Overall": float(analysis['match_percentage']),
                                    "Technical": float(analysis['technical_skills']['skill_match_percentage']),
                                    "Experience": float(analysis['experience_analysis']['experience_match_percentage']),
                                    "Education": float(analysis['education_analysis']['education_match']),
                                    "Cultural": float(analysis['cultural_fit']['alignment_score']),
                                    "Industry": float(analysis['industry_analysis']['industry_alignment'])
                                }
                                st.plotly_chart(create_radar_chart(skills_data), use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Detailed Analysis
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown('<div class="card">', unsafe_allow_html=True)
                                    st.subheader("✅ Strengths")
                                    st.write("Matching Skills:", ", ".join(analysis['technical_skills']['matching_skills']))
                                    st.write("Key Achievements:")
                                    for achievement in analysis['experience_analysis']['key_achievements']:
                                        st.write(f"• {achievement}")
                                    st.markdown('</div>', unsafe_allow_html=True)

                                with col2:
                                    st.markdown('<div class="card">', unsafe_allow_html=True)
                                    st.subheader("🎯 Areas for Improvement")
                                    st.write("Missing Skills:", ", ".join(analysis['technical_skills']['missing_skills']))
                                    st.write("Recommended Certifications:")
                                    for cert in analysis['education_analysis']['recommended_certifications']:
                                        st.write(f"• {cert}")
                                    st.markdown('</div>', unsafe_allow_html=True)

                            except Exception as e:
                                st.error(f"Error processing results: {str(e)}")
            else:
                st.warning("Please provide all required information: Job Title, Description, and Resume")

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 ATS Compatibility Check")
        if 'analysis' in locals():
            ats_score = int(analysis['ats_compatibility']['format_score'])
            st.progress(ats_score / 100)
            st.metric("ATS Score", f"{ats_score}%")

            col1, col2 = st.columns(2)
            with col1:
                st.write("📋 Format Issues:")
                for issue in analysis['ats_compatibility']['issues_found']:
                    st.warning(issue)
            with col2:
                st.write("💡 Suggestions:")
                for suggestion in analysis['ats_compatibility']['formatting_suggestions']:
                    st.info(suggestion)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Career Insights")
        if 'analysis' in locals():
            col1, col2 = st.columns(2)
            with col1:
                st.write("🎯 Industry Experience")
                for exp in analysis['industry_analysis']['relevant_experience']:
                    st.success(exp)
            with col2:
                st.write("🔄 Development Areas")
                for area in analysis['cultural_fit']['development_areas']:
                    st.info(area)
        st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("""
               <div style='text-align: center; margin-top: 2rem; padding: 1rem;'>
                   <p>Made with ❤️ byV56</p>
               </div>
           """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
