import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from utils import haversine

def solve_airbase_problem_gurobi(df_cities, R=150, B=120, P_min=40,
                                 lambda1=0.3, lambda2=0.5,
                                 n_zones=25, n_sites=12, k_min=2):
    """
    Solve the airbase location problem using Gurobi.

    FIXES:
    1. Ensure geographical spread by stratified sampling
    2. Fixed formatting issues
    """

    # Stratified sampling to ensure geographical distribution
    print("Performing stratified sampling to ensure geographical spread...")

    # Sort by latitude to ensure north-south distribution
    df_cities_sorted = df_cities.sort_values('lat').reset_index(drop=True)

    # Select zones with uniform distribution across latitude
    zone_indices = np.linspace(0, len(df_cities_sorted) - 1, n_zones, dtype=int)
    zones_df = df_cities_sorted.iloc[zone_indices].reset_index(drop=True)

    # Select sites with uniform distribution (different from zones)
    site_indices = np.linspace(0, len(df_cities_sorted) - 1, n_sites, dtype=int)
    # Shift indices to avoid overlap with zones
    site_indices = [(idx + len(df_cities_sorted) // (2 * n_sites)) % len(df_cities_sorted)
                    for idx in site_indices]
    sites_df = df_cities_sorted.iloc[site_indices].reset_index(drop=True)

    # Ensure no overlap between zones and sites
    common = pd.merge(zones_df, sites_df, on=['name', 'lat', 'lon'], how='inner')
    if not common.empty:
        # Replace overlapping sites with different cities
        for idx in range(len(common)):
            alt_idx = (site_indices[idx] + len(df_cities_sorted) // 3) % len(df_cities_sorted)
            sites_df.iloc[idx] = df_cities_sorted.iloc[alt_idx]

    zones = zones_df["name"].tolist()
    sites = sites_df["name"].tolist()

    coords_zones = zones_df[["lat", "lon"]].to_numpy()
    coords_sites = sites_df[["lat", "lon"]].to_numpy()

    print(f"Selected {len(zones)} zones and {len(sites)} potential sites")
    print(f"Zones latitude range: {coords_zones[:, 0].min():.2f} to {coords_zones[:, 0].max():.2f}")
    print(f"Sites latitude range: {coords_sites[:, 0].min():.2f} to {coords_sites[:, 0].max():.2f}")

    # ============================================================
    # Calculate Haversine distances
    # ============================================================

    print("Calculating distances...")
    dist = np.zeros((len(zones), len(sites)))
    for i in range(len(zones)):
        for j in range(len(sites)):
            dist[i, j] = haversine(
                coords_zones[i, 0], coords_zones[i, 1],
                coords_sites[j, 0], coords_sites[j, 1]
            )

    # Convert to dictionary for easier access
    dist_dict = {(zones[i], sites[j]): dist[i, j]
                 for i in range(len(zones)) for j in range(len(sites))}

    # ============================================================
    # Generate random parameters
    # ============================================================

    np.random.seed(42)
    cost = {sites[j]: np.random.randint(10, 40) for j in range(len(sites))}
    risk = {sites[j]: np.random.randint(1, 10) for j in range(len(sites))}
    capacity = {sites[j]: np.random.randint(5, 15) for j in range(len(sites))}

    # ============================================================
    # Create and solve Gurobi model
    # ============================================================

    print("Creating Gurobi model...")
    model = gp.Model("AirbaseLocation_Tunisia")

    # Decision variables
    x = model.addVars(sites, vtype=GRB.BINARY, name="x")
    y = model.addVars(zones, sites, vtype=GRB.BINARY, name="y")

    # Objective: weighted sum
    obj = (
            lambda1 * gp.quicksum(x[j] for j in sites) +
            lambda2 * gp.quicksum(cost[j] * x[j] for j in sites)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # 1. Coverage constraints
    print("Adding coverage constraints...")
    for i in zones:
        # Find all sites within radius R
        valid_sites = [j for j in sites if dist_dict[(i, j)] <= R]
        if valid_sites:
            model.addConstr(
                gp.quicksum(y[i, j] for j in valid_sites) >= 1,
                f"Coverage_{i}"
            )
        else:
            # If no site within R, find the closest one and force it to be selected
            closest_site = min(sites, key=lambda j: dist_dict[(i, j)])
            model.addConstr(x[closest_site] == 1, f"Force_{i}")

    # 2. Assignment only to open bases
    print("Adding assignment constraints...")
    for i in zones:
        for j in sites:
            model.addConstr(y[i, j] <= x[j], f"Assign_{i}_{j}")

    # 3. Budget constraint
    model.addConstr(
        gp.quicksum(cost[j] * x[j] for j in sites) <= B,
        "Budget"
    )

    # 4. Minimum total operational capacity constraint
    print("Adding minimum capacity constraint...")
    model.addConstr(
        gp.quicksum(capacity[j] * x[j] for j in sites) >= P_min,
        "Min_Capacity"
    )

    # 5. Geographical consistency: forbid assignment if distance > R
    print("Adding geographical constraints...")
    for i in zones:
        for j in sites:
            if dist_dict[(i, j)] > R:
                model.addConstr(y[i, j] == 0, f"No_assign_{i}_{j}")

    # 6. Minimum number of bases constraint
    print("Adding minimum bases constraint...")
    model.addConstr(
        gp.quicksum(x[j] for j in sites) >= k_min,
        "Min_Bases"
    )

    # 7. Maximum zones per base constraint
    print("Adding maximum zones per base constraint...")
    max_zones_per_base = min(10, len(zones) // 2)
    for j in sites:
        model.addConstr(
            gp.quicksum(y[i, j] for i in zones) <= max_zones_per_base * x[j],
            f"Max_Zones_{j}"
        )

    # 8. Minimum distance between selected bases
    print("Adding minimum distance between bases constraint...")
    min_base_distance = R * 0.7  # Bases must be at least 70% of coverage radius apart

    # Pre-calculate distances between all sites
    site_distances = np.zeros((len(sites), len(sites)))
    for j1 in range(len(sites)):
        for j2 in range(len(sites)):
            if j1 != j2:
                site_distances[j1, j2] = haversine(
                    coords_sites[j1, 0], coords_sites[j1, 1],
                    coords_sites[j2, 0], coords_sites[j2, 1]
                )

    # Add constraints for sites that are too close
    for j1 in range(len(sites)):
        for j2 in range(j1 + 1, len(sites)):
            if site_distances[j1, j2] < min_base_distance:
                model.addConstr(x[sites[j1]] + x[sites[j2]] <= 1,
                                f"Min_Dist_{j1}_{j2}")

    # 9. Regional coverage constraint
    print("Adding regional coverage constraints...")
    regions = {
        "North": (36.0, 37.6),
        "Central": (34.0, 36.0),
        "South": (30.2, 34.0)
    }

    region_vars = {}
    for region_name, (lat_min, lat_max) in regions.items():
        sites_in_region = [
            sites[j] for j in range(len(sites))
            if lat_min <= coords_sites[j, 0] <= lat_max
        ]

        if sites_in_region:
            region_var = model.addVar(vtype=GRB.BINARY, name=f"region_{region_name}")
            region_vars[region_name] = region_var

            # Link region variable to site selection
            model.addConstr(
                gp.quicksum(x[j] for j in sites_in_region) >= region_var,
                f"Region_{region_name}_cover"
            )

    # Require at least 2 out of 3 regions to have bases
    if len(region_vars) >= 2:
        model.addConstr(
            gp.quicksum(region_vars.values()) >= 2,
            "Min_Regions"
        )

    # Set parameters
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = 300
    model.Params.MIPGap = 0.01

    # Solve
    print("Solving model...")
    model.optimize()

    # ============================================================
    # Results analysis
    # ============================================================

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.status == GRB.OPTIMAL:
            print(f"\n{'=' * 60}")
            print("OPTIMAL SOLUTION FOUND")
            print(f"{'=' * 60}")
        else:
            print(f"\n{'=' * 60}")
            print("TIME LIMIT REACHED - Best solution found")
            print(f"{'=' * 60}")

        selected_sites = [j for j in sites if x[j].X > 0.5]

        print(f"\nSELECTED SITES ({len(selected_sites)}):")
        for j in selected_sites:
            idx = sites.index(j)
            lat, lon = coords_sites[idx, 0], coords_sites[idx, 1]

            # Determine region
            if lat >= 36.0:
                region = "North"
            elif lat >= 34.0:
                region = "Central"
            else:
                region = "South"

            print(f"  {j}")
            print(f"    Region: {region}, Location: ({lat:.2f}, {lon:.2f})")
            print(f"    Cost: {cost[j]}, Risk: {risk[j]}, Capacity: {capacity[j]}")

        # Calculate metrics
        total_cost = sum(cost[j] * x[j].X for j in sites)
        total_risk = sum(risk[j] * x[j].X for j in sites)
        total_capacity = sum(capacity[j] * x[j].X for j in sites)
        num_bases = sum(x[j].X for j in sites)

        print(f"\n{'=' * 60}")
        print("PERFORMANCE METRICS:")
        print(f"{'=' * 60}")
        print(f"Number of bases opened: {num_bases}")
        print(f"Total cost: {total_cost:.2f} / {B} ({total_cost / B * 100:.1f}% of budget)")
        print(f"Total risk: {total_risk:.2f}")
        print(f"Total capacity: {total_capacity:.2f} (min required: {P_min})")
        print(f"Objective value (weighted): {model.ObjVal:.2f}")

        # Coverage analysis
        uncovered_zones = []
        zone_coverage_count = {}

        for i in zones:
            covering_bases = [j for j in sites if y[i, j].X > 0.5]
            zone_coverage_count[i] = len(covering_bases)
            if len(covering_bases) == 0:
                uncovered_zones.append(i)

        if uncovered_zones:
            print(f"\nWARNING: {len(uncovered_zones)} zones are NOT covered!")
            for zone in uncovered_zones[:5]:
                print(f"  - {zone}")
        else:
            print(f"\nAll zones are covered by at least one base.")

        # Show coverage statistics
        avg_coverage = np.mean(list(zone_coverage_count.values())) if zone_coverage_count else 0
        min_coverage = min(zone_coverage_count.values()) if zone_coverage_count else 0
        max_coverage = max(zone_coverage_count.values()) if zone_coverage_count else 0

        print(f"\nCOVERAGE STATISTICS:")
        print(f"  Average bases per zone: {avg_coverage:.2f}")
        print(f"  Minimum bases covering a zone: {min_coverage}")
        print(f"  Maximum bases covering a zone: {max_coverage}")

        # Show geographical distribution
        print(f"\nGEOGRAPHICAL DISTRIBUTION:")
        region_counts = {"North": 0, "Central": 0, "South": 0}
        for j in selected_sites:
            idx = sites.index(j)
            lat = coords_sites[idx, 0]
            if lat >= 36.0:
                region_counts["North"] += 1
            elif lat >= 34.0:
                region_counts["Central"] += 1
            else:
                region_counts["South"] += 1

        for region, count in region_counts.items():
            print(f"  {region}: {count} base(s)")

        # Create results DataFrame
        results_df = pd.DataFrame({
            "Site": sites,
            "Opened": [int(x[j].X > 0.5) for j in sites],
            "Cost": [cost[j] for j in sites],
            "Risk": [risk[j] for j in sites],
            "Capacity": [capacity[j] for j in sites],
            "Latitude": [coords_sites[j, 0] for j in range(len(sites))],
            "Longitude": [coords_sites[j, 1] for j in range(len(sites))]
        })

        # Create distance matrix DataFrame - ensure all values are numeric
        distance_matrix = pd.DataFrame(dist, index=zones, columns=sites)

        print(f"\n{'=' * 60}")
        print("RESULTS TABLE:")
        print(f"{'=' * 60}")
        print(results_df[['Site', 'Opened', 'Cost', 'Risk', 'Capacity']].to_string())

        return {
            "model": model,
            "x_vars": x,
            "y_vars": y,
            "selected_sites": selected_sites,
            "zones": zones,
            "sites": sites,
            "coords_zones": coords_zones,
            "coords_sites": coords_sites,
            "dist_dict": dist_dict,
            "distance_matrix": distance_matrix,
            "results_df": results_df,
            "cost_dict": cost,
            "risk_dict": risk,
            "capacity_dict": capacity,
            "zone_coverage_count": zone_coverage_count,
            "metrics": {
                "total_cost": total_cost,
                "total_risk": total_risk,
                "total_capacity": total_capacity,
                "num_bases": num_bases,
                "objective": model.ObjVal,
                "avg_coverage": avg_coverage,
                "min_coverage": min_coverage,
                "max_coverage": max_coverage
            },
            "parameters": {
                "R": R,
                "B": B,
                "P_min": P_min,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "k_min": k_min
            }
        }

    else:
        print(f"\nOptimization failed with status: {model.status}")
        return None