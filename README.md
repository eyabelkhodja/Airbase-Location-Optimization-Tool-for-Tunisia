# ✈️ Airbase Location Optimization Tool for Tunisia

A Python-based decision support tool that optimizes the placement of airbases across Tunisia using mathematical programming. The application combines a PyQt5 graphical interface with Gurobi optimization to select the best locations based on coverage, cost, capacity, and risk criteria.

---

## 🚀 Features

- **Data Acquisition**: Retrieves real municipality data from the Tunisian INS API (with fallback to sample data).
- **Geographical Spread**: Stratified sampling ensures a balanced north-south distribution of potential zones and sites.
- **Gurobi Optimization**: Solves a mixed-integer programming model that accounts for:
  - Coverage radius constraints  
  - Budget and capacity limits  
  - Minimum number of bases  
  - Geographical consistency (assignment only within coverage radius)  
  - Maximum zones per base  
  - Minimum distance between bases  
  - Regional coverage requirements  

- **Interactive GUI**:
  - Parameter tuning (coverage radius, budget, capacity, weights, etc.)
  - Visual map with coverage ellipses
  - Distance matrix table (zones to sites)
  - City data viewer
  - Base cost/risk/capacity summary with filtering and sorting

- **Results Display**:
  - Selected bases with coordinates, cost, risk, capacity
  - Performance metrics (cost usage, coverage statistics)
  - Summary in the interface and console

---

## ⚙️ Requirements

- Python 3.7 or higher  
- **Gurobi Optimizer** (license required; free academic licenses available)

### Python packages:
- `numpy`
- `pandas`
- `matplotlib`
- `requests`
- `PyQt5`

### Install dependencies:
```bash
pip install numpy pandas matplotlib requests PyQt5
```

Gurobi must be installed separately. Visit the official Gurobi website and follow the installation instructions.  
After installation, install the Python interface if needed:

```bash
pip install gurobipy
```

⚠️ **Important Warning**  
The **Gurobi license version must match the installed `gurobipy` version**.  
If they are incompatible, you may encounter errors such as license validation failure or solver initialization issues.  
Always ensure:
- Your Gurobi installation version = your `gurobipy` package version  
- Your license is valid for that specific version  

---

## 📦 Installation

1. Clone the repository or download the source code:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

2. Ensure all files are in the same directory:
- `utils.py`
- `data.py`
- `model.py`
- `gui.py`
- `main.py`

3. Install all dependencies.

4. Run the application:

```bash
python main.py
```

---

## ▶️ Usage

1. Launch the application.
2. Data will load automatically from the INS API (or fallback to sample data).
3. Adjust parameters in the Control Panel:
   - Coverage radius (km)
   - Budget
   - Minimum total capacity
   - Number of zones and potential sites
   - Minimum number of bases
   - Weights (number of bases vs cost)

4. Click **Run Optimization**.
5. Explore results in the tabs:

- **Visualization**: Map with zones, sites, selected bases, and coverage ellipses  
- **Distance Matrix**: Distances from zones to sites with filtering  
- **City Data**: Municipality list  
- **Base Costs**: Sortable/filterable base characteristics  

6. Use **Reload City Data** to refresh data anytime.

---

## 🗂️ Project Structure

```
├── utils.py          # Haversine distance function
├── data.py           # Fetch Tunisian municipalities (API or sample)
├── model.py          # Gurobi optimization model
├── gui.py            # PyQt5 graphical interface
└── main.py           # Application entry point
```

---

## 🧠 How It Works

1. Municipality data is stored in a Pandas DataFrame (`name`, `lat`, `lon`).
2. Cities are split into:
   - **Zones** (demand points)
   - **Potential airbase sites**

3. Distances are computed using the Haversine formula.

4. A **Mixed-Integer Linear Program (MILP)** is solved using Gurobi:

- **Decision variables**:
  - `x_j`: 1 if a base is built at site *j*
  - `y_{i,j}`: 1 if zone *i* is assigned to site *j*

- **Objective**:
  Minimize:
  ```
  λ1 × (number of bases) + λ2 × (total cost)
  ```

- **Constraints**:
  - Coverage radius
  - Budget limit
  - Capacity requirements
  - Assignment validity
  - Distance restrictions
  - Regional coverage rules

5. Results are displayed in the GUI and printed in the console.

---

## ⚡ Customization

- Modify `regions` in `model.py` for regional constraints
- Change random seed in `model.py` for reproducibility
- Adjust number of zones/sites or sampling strategy

---

## 🛠️ Troubleshooting

- **Gurobi license error**  
  Ensure Gurobi is installed correctly and a license is active:
  ```bash
  grbprobe
  ```

- **Version mismatch error**  
  If you see errors related to licensing or solver initialization, verify that:
  - `gurobipy` version matches the installed Gurobi version
  - Your license supports that version

- **API failure**  
  Falls back to sample data automatically.

- **No feasible solution**  
  Try:
  - Increasing budget  
  - Lowering capacity requirements  
  - Allowing more zones/sites  

---

## 📜 License

This project is provided for educational and research purposes. Use at your own discretion.

---

## 🙏 Acknowledgments

- INS (Tunisian National Institute of Statistics) for municipality data  
- Gurobi Optimization for the MIP solver  

---
