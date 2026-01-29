# F1 Outcome Lab

Formula 1 race outcome prediction model and API.

## Project Structure

- `src/f1outcome/` - Main package
  - `config.py` - Configuration settings
  - `data/` - Data collection and processing
  - `models/` - Prediction models
  - `api/` - FastAPI application
- `scripts/` - Training and data processing scripts
- `data/` - Raw and processed data
- `artifacts/` - Model artifacts and outputs

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Build Dataset

```bash
python scripts/build_dataset.py
```

### Train Model

```bash
python scripts/train_ranker.py
```

### Run API

```bash
uvicorn src.f1outcome.api.app:app --reload
```

## License

MIT
