# UNLEASH Log Parser

This project implements a UNLEASH-inspired log parsing system using entropy-based sampling and RoBERTa.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/unleash-log-parser.git
cd unleash-log-parser
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets
Download datasets from LogHub and place them inside the data/ folder.
Make sure each dataset contains:
- Content
- Event
- Template

### 5. Run Pipeline
```bash
python scripts/run_pipeline.py
```
EventId
