# StockAI - Comprehensive Stock Analysis Platform

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Required API Keys](#required-api-keys)
- [How to Run](#how-to-run)
- [Execution Flow](#execution-flow)
- [Dashboard Interface](#dashboard-interface)
- [Components](#components)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Future Development](#future-development)

## Overview

StockAI is an advanced stock analysis platform that combines AI-powered research, technical analysis, fundamental analysis, and machine learning predictions to provide comprehensive investment insights. The system leverages multiple AI agents powered by different LLMs (Large Language Models) and specialized financial tools to generate detailed investment reports and actionable recommendations.

## Features

- **AI-Driven Market Research**: Comprehensive market research using Gemini 2.0 Flash LLM
- **Technical Analysis**: Moving averages, support/resistance levels, momentum indicators, and chart patterns
- **Fundamental Analysis**: Financial ratios, growth metrics, and valuation models
- **Reddit Sentiment Analysis**: Social media sentiment tracking and analysis
- **Financial News Integration**: Latest news from Alpha Vantage or Yahoo Finance
- **Parallel Execution**: Multi-threaded analysis for faster results
- **Interactive Dashboard**: Clean, intuitive Streamlit interface
- **Detailed Reports**: Comprehensive investment reports with specific recommendations
- **Usage Analytics**: Detailed token usage tracking for LLM optimization

## Technical Architecture

StockAI uses a multi-tier architecture:

1. **Data Layer**: Tools that connect to financial APIs and data sources (Yahoo Finance, Alpha Vantage, Reddit, web search)
2. **Analysis Layer**: Direct technical and fundamental analysis tools + AI agents for research and reporting
3. **Orchestration Layer**: `FinancialAdvisor` class that manages agents, tasks, and workflow
4. **Presentation Layer**: Streamlit web interface for user interaction

The system can operate in both sequential and parallel modes:
- **Sequential Mode**: Tasks are executed one after another
- **Parallel Mode**: Independent tasks run concurrently using Python's threading module

## Directory Structure

```
stockai/
├── main.py                 # Streamlit app entry point
├── crew.py                 # Financial Advisor implementation
├── tools/                  # Financial analysis tools
│   ├── yf_tech_analysis.py            # Technical analysis tool
│   ├── yf_fundamental_analysis.py     # Fundamental analysis tool
│   ├── sentiment_analysis.py          # Reddit sentiment analysis
│   ├── yahoo_finance_tool.py          # Yahoo Finance news tool
│   └── AlphaVantage_finance_tool.py   # Alpha Vantage market data tool
├── config/                 # Configuration files
│   ├── agents.yaml         # Agent definitions
│   └── tasks.yaml          # Task descriptions
├── results/                # Analysis output directory
│   └── [STOCK_SYMBOL]/     # Stock-specific results
├── models/                 # (Optional) ML prediction models
│   └── lstm_predictor.py   # (If implemented) LSTM prediction
├── .env                    # Environment variables
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SoumyaeCodes/StockLensAI.git
cd LLM Dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root with the following variables:

```
# Required API Keys
SERPER_API_KEY=your_serper_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
GEMINI_API_KEY=your_gemini_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Optional - Multiple Groq accounts for advanced parallelization
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_1=your_first_groq_key
GROQ_API_KEY_2=your_second_groq_key

# Configuration Options
USE_PARALLEL_EXECUTION=true
SAVE_INTERIM_RESULTS=true
DEBUG_MODE=false
```

## Required API Keys

1. **SerperDev API Key** (`SERPER_API_KEY`)
   - Used for: Web search functionality
   - Get from: [SerperDev](https://serper.dev)
   - Pricing: Free tier available with limited daily searches

2. **Reddit API Credentials** (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`)
   - Used for: Sentiment analysis on Reddit posts
   - Get from: [Reddit Developer Portal](https://www.reddit.com/prefs/apps)
   - Setup: Create an application, select "script" type

3. **Google Gemini API Key** (`GEMINI_API_KEY`)
   - Used for: Research and report generation agents
   - Get from: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Pricing: Free tier available with monthly quota

4. **Alpha Vantage API Key** (`ALPHA_VANTAGE_API_KEY`)
   - Used for: Financial news and market data
   - Get from: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Pricing: Free tier available with limited daily API calls

5. **Groq API Key** (`GROQ_API_KEY`)
   - Used for: (Optional) Alternative LLM processing
   - Get from: [Groq Console](https://console.groq.com)
   - Pricing: Various tiers available

## How to Run

### Starting the Dashboard

1. Ensure your virtual environment is activated
2. Run the Streamlit application:
```bash
streamlit run main.py
```
3. The dashboard will open in your default web browser at `http://localhost:8501`

### Command Line Usage

For batch processing or API integration, you can use the module directly:

```python
from crew import run_analysis

# Run analysis for a specific stock
result = run_analysis("AAPL")

# Access the report
print(result['report'])
```

## Execution Flow

The financial analysis code executes in the following sequence:

1. **Initialization**:
   - Load environment variables from `.env`
   - Initialize LLM configurations
   - Load agent and task definitions from YAML files
   - Initialize required tools (SerperDev, Reddit, etc.)

2. **Research Phase**:
   - Create research agent using Gemini 2.0 Flash LLM
   - Execute research task with specified stock symbol
   - Save interim research results to disk

3. **Analysis Phase** (Sequential or Parallel):
   - **Sequential Mode**: Execute technical analysis, then fundamental analysis
   - **Parallel Mode**: Execute technical and fundamental analyses concurrently
   - Both analyses directly use tools without LLM involvement
   - Save interim analysis results to disk

4. **Reporting Phase**:
   - Combine all previous analysis results
   - Clean and format text for LLM processing
   - Create reporter agent using Gemini 2.0 Flash LLM
   - Generate comprehensive investment report
   - Save final report to disk

5. **Result Delivery**:
   - Return structured results to caller or display in UI
   - Track and report token usage statistics

### Detailed Component Workflow

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────────┐
│  Research Task │     │ Technical Task  │     │ Fundamental Task  │
│  (Gemini LLM)  │────>│ (Direct Tool)   │────>│ (Direct Tool)     │
└────────────────┘     └─────────────────┘     └───────────────────┘
        │                      │                         │
        │                      │                         │
        ▼                      ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Report Task                               │
│                       (Gemini LLM)                               │
└─────────────────────────────────────────────────────────────────┘
```

**Parallel Execution Workflow**:

```
┌────────────────┐
│  Research Task │
│  (Gemini LLM)  │
└───────┬────────┘
        │
        ▼
┌───────────────────────┐
│      Split Results    │
└───────────────────────┘
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
┌─────────────────┐    ┌───────────────────┐
│ Technical Task  │    │ Fundamental Task  │
│ (Direct Tool)   │    │ (Direct Tool)     │
└────────┬────────┘    └─────────┬─────────┘
         │                       │
         │                       │
         ▼                       ▼
┌───────────────────────────────────────────┐
│              Join Results                 │
└───────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│              Report Task                  │
│             (Gemini LLM)                  │
└───────────────────────────────────────────┘
```

## Dashboard Interface

The Streamlit dashboard consists of the following components:

### Input Section
- **Stock Symbol Input**: Text box for entering ticker symbols
- **Analysis Options**: Checkboxes for selecting which analyses to run
- **Execution Mode**: Toggle for sequential vs. parallel execution
- **Analyze Button**: Starts the analysis process

### Results Section
- **Tabbed Interface**: Contains separate tabs for:
  - Report Tab: Final comprehensive investment report
  - Research Tab: Detailed market research findings
  - Technical Tab: Technical analysis results and indicators
  - Fundamental Tab: Financial metrics and valuation analysis
  - Sentiment Tab: Reddit and social media sentiment analysis

### Visualization Section
- **Price Charts**: Historical price visualization
- **Technical Indicators**: Visual representation of key indicators
- **Prediction Charts**: (If implemented) LSTM prediction visualization

### Export Options
- **Download Report**: Save report as markdown, PDF, or HTML
- **Save Analysis**: Save complete analysis to file
- **Share Results**: (If implemented) Share via link or email

## Components

### 1. FinancialAdvisor Class (`crew.py`)

The core orchestration class that manages:
- Agent creation and configuration
- Task execution and sequencing
- Tool integration and data flow
- Results formatting and storage

```python
@CrewBase
class FinancialAdvisor:
    # Initialize with configuration
    def __init__(self, agents_config, tasks_config, stock_symbol):
        # Configuration and setup
        
    # Define agents
    @agent
    def researcher(self) -> Agent:
        # Research agent definition
        
    # Define tasks
    @task
    def research_task(self) -> Task:
        # Research task definition
        
    # Analysis methods
    def execute_technical_analysis_directly(self):
        # Direct tool execution for technical analysis
        
    def execute_fundamental_analysis_directly(self):
        # Direct tool execution for fundamental analysis
        
    # Execution methods
    def run_sequential_analysis(self):
        # Sequential execution workflow
        
    def run_parallel_analysis(self):
        # Parallel execution workflow
        
    # Helper methods
    def _save_interim_result(self, task_name, result):
        # Save results to disk
```

### 2. Analysis Tools (in `tools/` directory)

Specialized tools for different analysis aspects:

- **YFinanceTechnicalAnalysisTool**: Analyzes price patterns, moving averages, momentum indicators
- **YFinanceFundamentalAnalysisTool**: Analyzes financial statements, ratios, and valuations
- **RedditSentimentAnalysisTool**: Analyzes sentiment from Reddit posts
- **YahooFinanceNewsTool**: Retrieves latest news from Yahoo Finance
- **AlphaVantageNewsTool**: Retrieves news and market data from Alpha Vantage

### 3. LLM Configuration

Multiple LLM configurations for different tasks:

```python
# Research LLM - factual, precise
gemini_research_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_api_key,
    temperature=0.2,  # Lower temperature for more factual outputs
)

# Reporter LLM - balanced, creative
gemini_reporter_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_api_key,
    temperature=0.5,  # Higher temperature for more creative synthesis
)
```

### 4. Data Models

Pydantic models for structured data:

```python
class ResearchReport(BaseModel):
    researchreport: str

class TechnicalAnalysisReport(BaseModel):
    techsummary: str

class FundamentalAnalyisReport(BaseModel):
    summary: str

class FinancialReport(BaseModel):
    report: str
```

## Performance Optimization

The system uses several optimization techniques:

1. **Parallel Execution**: Multi-threaded processing for independent tasks
2. **Caching**: Results are saved to disk for potential reuse
3. **LLM Temperature Settings**: Different temperature settings for different tasks
4. **Structured Output**: Pydantic models ensure consistent output structure
5. **Error Handling**: Robust error handling with fallback mechanisms

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Symptom: "API key not found" or authentication errors
   - Solution: Verify all API keys in your `.env` file and check API provider dashboards for quota limits

2. **Module Not Found Errors**
   - Symptom: "ModuleNotFoundError" when starting the application
   - Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

3. **Memory Errors**
   - Symptom: "MemoryError" or application crashes
   - Solution: Reduce the context size for LLMs or optimize memory usage in tools

### Debugging

Enable debug mode by setting `DEBUG_MODE=true` in your `.env` file or by passing `verbose=True` to appropriate components:

```python
# Enable verbose logging for agents
agent = Agent(
    role="Researcher",
    verbose=True,  # Enable detailed logging
    # Other parameters...
)
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Yahoo Finance for financial data
- Alpha Vantage for market data and news
- Reddit API for sentiment analysis
- SerperDev for web search capabilities
- Google Gemini for LLM capabilities
- CrewAI for agent orchestration framework
