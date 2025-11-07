# Auto-Preprocess LLM Project

This project provides an automated data preprocessing and analysis pipeline using Huawei Cloud LLM services and Streamlit for the interface.

## Project Structure

```
├── app.py                 # Main Streamlit application
├── preprocessing/         # Core preprocessing modules
│   ├── llm_advisor.py    # LLM integration for analysis
│   ├── pipeline.py       # Data preprocessing pipeline
│   ├── report.py        # Report generation
│   ├── stats.py         # Statistical analysis
│   └── transformers.py  # Data transformation utilities
├── assets/              # Static assets and resources
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment specification
├── Dockerfile          # Container definition
└── docker-compose.yml  # Container orchestration
```

## Prerequisites

- Docker and Docker Compose installed
- Access to Huawei Cloud LLM services
- API credentials for Huawei Cloud

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Configure Environment**
   
   Create a `.env` file in the root directory with your Huawei Cloud credentials:
   ```env
   HUAWEI_LLM_ENDPOINT=<your-endpoint>
   HUAWEI_LLM_API_KEY=<your-api-key>
   LLM_MODEL_NAME=qwen3-32b
   ```

3. **Build and Run with Docker**
   ```bash
   docker-compose up --build
   ```
   The application will be available at `http://localhost:8501`

## Using without Docker

If you prefer to run without Docker:

1. **Create a Python Environment**
   ```bash
   # Using conda
   conda env create -f environment.yml
   conda activate auto-preprocess-llm

   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Key Features

- Automated data preprocessing pipeline
- LLM-powered data analysis and recommendations
- Statistical analysis and reporting
- Interactive Streamlit interface
- Docker containerization for easy deployment
- Huawei Cloud LLM integration

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| LLM_MODEL_NAME | Name of the LLM model | qwen3-32b |
| HUAWEI_LLM_ENDPOINT | Huawei Cloud LLM endpoint URL | - |
| HUAWEI_LLM_API_KEY | Huawei Cloud API key | - |

## Docker Configuration

The Docker setup includes:
- Python 3.11 base image
- Essential system dependencies
- Streamlit configuration for container deployment
- Volume mapping for local development
- Automatic port mapping (8501)

## Development Notes

- The application uses Streamlit for the web interface
- Data preprocessing modules are in the `preprocessing/` directory
- Statistical analysis and transformations are modular
- LLM integration is handled through the `llm_advisor.py` module

## Troubleshooting

1. **Docker Issues**
   - Ensure Docker daemon is running
   - Check port 8501 is not in use
   - Verify Docker has sufficient resources

2. **API Connection**
   - Verify Huawei Cloud credentials
   - Check network connectivity
   - Ensure endpoint URLs are correct

3. **Data Processing**
   - Check input data format
   - Verify memory requirements
   - Review preprocessing logs

## Best Practices

- Keep the `.env` file secure and never commit it
- Update dependencies regularly
- Monitor API usage and costs
- Back up your data regularly
- Test locally before deployment

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Huawei Cloud documentation
3. Contact the development team

## License

[Your License Information]