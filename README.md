# Resume Analyzer Pro 📊

Resume Analyzer Pro is an AI-powered resume analysis tool that helps job seekers optimize their resumes for specific job positions. Built with Streamlit and powered by Google's Gemini AI, this application provides comprehensive insights, ATS compatibility checks, and personalized recommendations.

## 🌟 Features

- **AI-Powered Analysis**: Utilizes Google's Gemini AI for intelligent resume parsing and analysis
- **Job Matching**: Compares resumes against specific job descriptions
- **Skill Gap Analysis**: Identifies matching and missing skills
- **Interactive Visualizations**: 
  - Radar charts for skills assessment
  - Timeline visualization for experience
  - Word clouds for key terms
- **ATS Compatibility Check**: Ensures resume formatting meets ATS requirements
- **Career Insights**: Provides industry-specific recommendations and development areas
- **Multiple Export Options**: Support for PDF, DOCX, and JSON formats
- **Dynamic Theme Support**: Light and dark mode options

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Google Cloud API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-analyzer-pro.git
cd resume-analyzer-pro
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your Google API key:
```
Google_api_key=your_api_key_here
```

### Required Dependencies

```txt
streamlit
google-generativeai
python-docx2txt
PyPDF2
python-dotenv
plotly
wordcloud
matplotlib
pandas
numpy
streamlit-lottie
streamlit-option-menu
streamlit-extras
```

## 💻 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application through your web browser at `http://localhost:8501`

3. Upload your resume (PDF or DOCX format)

4. Enter the job title and description

5. Click "Analyze Resume" to get comprehensive insights

## 🔍 Features Breakdown

### Resume Analysis
- Match percentage calculation
- Technical skills assessment
- Experience analysis
- Education evaluation
- Cultural fit assessment

### ATS Compatibility
- Format score
- Issue identification
- Formatting recommendations

### Career Insights
- Industry alignment analysis
- Development areas identification
- Skill gap recommendations
- Certification suggestions

## 🎨 Customization

### Export Options
Analysis results can be exported in multiple formats:
- PDF Report
- DOCX Document
- JSON Data

## 🛠️ Technical Architecture

The application is built using:
- **Frontend**: Streamlit
- **AI Engine**: Google Gemini AI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Document Processing**: PyPDF2, docx2txt

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Generative AI for providing the AI capabilities
- Streamlit for the wonderful web framework
- All contributors and users of the application

## 📧 Contact

For any queries or suggestions, please open an issue in the repository.

---
Made with ❤️ by V56
