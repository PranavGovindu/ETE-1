# End-to-End Machine Learning Project (ETE-1)

A comprehensive machine learning platform that integrates data versioning, model lifecycle management, production deployment, and real-time monitoring. Built with scalability and observability at its core, this platform provides end-to-end ML workflow management from data processing to model serving.

## Features

- Data versioning and management using DVC
- Machine learning model training and tracking with MLflow
- FastAPI-based REST API for model serving
- Docker containerization for easy deployment
- Comprehensive logging and monitoring
- S3 integration for data storage
- Automated testing and CI/CD pipeline
- Real-time metrics visualization with Grafana
- Prometheus-based metrics collection

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- AWS CLI configured (for S3 access)
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ETE-1
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the project in development mode:
```bash
pip install -e .
```

## Project Structure

```
├── API/                 # FastAPI application
├── data/               # Data directory (DVC managed)
├── docs/               # Documentation
├── logs/               # Application logs
├── models/             # Trained model artifacts
├── monitoring/         # Monitoring configuration
│   ├── grafana/       # Grafana dashboards
│   └── prometheus/    # Prometheus configuration
├── notebooks/          # Jupyter notebooks
├── references/         # Reference materials
├── reports/            # Generated reports
├── src/                # Source code
│   ├── data/          # Data processing modules
│   ├── features/      # Feature engineering
│   ├── models/        # Model training
│   └── visualization/ # Visualization utilities
├── tests/             # Test files
├── .dvc/              # DVC configuration
├── .github/           # GitHub workflows
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker services orchestration
├── dvc.yaml          # DVC pipeline configuration
├── Makefile          # Project commands
├── params.yaml       # Project parameters
└── requirements.txt  # Project dependencies
```

## Usage

### Data Management

1. Pull data from DVC:
```bash
dvc pull
```

2. Run the DVC pipeline:
```bash
dvc repro
```

### Model Training

1. Train the model:
```bash
make train
```

2. Track experiments with MLflow:
```bash
mlflow ui
```

### API Deployment

1. Start the FastAPI server:
```bash
uvicorn API.main:app --reload
```

2. Access the API documentation at `http://localhost:5000/docs`

### Monitoring Stack

1. Start the monitoring services:
```bash
docker-compose up -d
```

2. Access the monitoring interfaces:
   - Grafana Dashboard: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - API Metrics: http://localhost:5000/metrics

The monitoring stack provides:
- Real-time request rate monitoring
- Latency tracking
- Prediction distribution analysis
- Error rate monitoring
- Resource utilization metrics

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ete-ml .
```

2. Run the container:
```bash
docker run -p 5000:5000 ete-ml
```

## Testing

Run tests using:
```bash
make test
```

## Monitoring

The platform includes comprehensive monitoring capabilities:

- Application logs are stored in the `logs/` directory
- MLflow tracking server for experiment monitoring
- API metrics available through FastAPI's built-in monitoring
- Grafana dashboards for real-time visualization
- Prometheus for metrics collection and storage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


