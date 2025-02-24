# Graph and Spatial Region Partitioning

This project implements and extends the PRRP (P-Regionalization through Recursive Partitioning) algorithm from the paper "Statistical Inference for Spatial Regionalization" (SIGSPATIAL 2023). The implementation includes both the original spatial regionalization algorithm and a new graph partitioning module.

## Project Structure

- `src/spatial/`: Original PRRP implementation for spatial data
- `src/graph/`: New graph partitioning module
- `src/experiments/`: Experimental evaluation code
- `data/`: Input datasets
- `results/`: Experimental results
- `notebooks/`: Analysis notebooks

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Spatial Regionalization
```bash
python src/experiments/spatial_experiments.py
```

### Graph Partitioning
```bash
python src/experiments/graph_experiments.py
```

## Project Components

### Spatial Module
- Data loading and preprocessing
- Region building with three phases:
  - Region growing
  - Region merging
  - Region splitting
- Visualization and evaluation

### Graph Module
- Graph data structure handling
- Partition generation with size constraints
- Connectivity maintenance
- Quality metrics:
  - Partition connectivity
  - Cut edge count
  - Size balance

## Evaluation Metrics

1. Runtime Performance
2. Solution Quality:
   - Connectivity validation
   - Balance measures
   - Cut size
3. Algorithm Success Rate

## References

1. Hussah Alrashid, Amr Magdy, Sergio Rey: Statistical Inference for Spatial Regionalization. In SIGSPATIAL 2023.

## Results

Detailed experimental results can be found in the `results/` directory and analysis notebooks.