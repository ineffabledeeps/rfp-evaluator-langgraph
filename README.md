# RFP Evaluator - Agentic Workflow

An intelligent Request for Proposal (RFP) evaluation system powered by LangGraph and OpenAI. This application uses an agentic workflow to automatically evaluate supplier proposals against predefined criteria, rank suppliers, and generate detailed scorecards.

## Features

- **Intelligent RFP Evaluation**: AI-powered agent evaluates supplier proposals using LangGraph
- **Streamlit UI**: User-friendly web interface for managing evaluations
- **Supplier Ranking**: Automatic ranking and leaderboard of suppliers
- **Detailed Scorecards**: Generate comprehensive evaluation reports and JSON exports
- **Database Storage**: Persistent storage of evaluation results and supplier metadata
- **PDF Processing**: Extract and process PDF RFP documents

## Requirements

- Python 3.9+
- OpenAI API key

## Installation

1. **Clone the repository**:
   ```bash
   cd rfp-evaluator
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv\Scripts\activate  # On Windows
   # or: source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root and add:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`
   - The application should automatically open in your default browser

3. **Using the Application**:
   - **Criteria**: Set up evaluation criteria for supplier assessment
   - **New Evaluation**: Upload RFP documents and run the evaluation pipeline
   - **Leaderboard**: View results, detailed scorecards, and download evaluation data

## Project Structure

- `app.py` - Streamlit web interface and main application logic
- `db.py` - Database operations and storage management
- `pipeline.py` - Agentic workflow pipeline using LangGraph
- `requirements.txt` - Python package dependencies

## License

See LICENSE file for details.
