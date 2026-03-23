import numpy as np
import pandas as pd
import requests

def fetch_tunisian_municipalities():
    """Fetch Tunisian municipalities from the API."""
    print("Fetching Tunisian municipalities from API...")
    BASE_URL = "https://tn-municipality-api.vercel.app"

    try:
        resp = requests.get(f"{BASE_URL}/api/municipalities", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        municipalities = []
        for m in data:
            name = m.get("Name")
            for d in m.get("Delegations", []):
                lat = d.get("Latitude")
                lon = d.get("Longitude")
                if lat is None or lon is None:
                    continue
                try:
                    lat_float = float(lat)
                    lon_float = float(lon)
                except (ValueError, TypeError):
                    continue
                municipalities.append({
                    "name": name + " / " + d.get("Name"),
                    "lat": lat_float,
                    "lon": lon_float
                })

        df_cities = pd.DataFrame(municipalities)
        if df_cities.empty:
            raise ValueError("No cities retrieved from INS API.")

        print(f"Retrieved {len(df_cities)} municipalities")
        return df_cities

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        print("Using sample data instead...")
        return create_sample_data()


def create_sample_data():
    """Create sample data if API fails."""
    np.random.seed(42)

    tunisia_lat = (30.2, 37.6)  # From south to north
    tunisia_lon = (7.5, 11.6)   # From west to east

    n_samples = 70
    names = [f"Municipality_{i}" for i in range(n_samples)]

    lats = []
    lons = []

    # North region (more populated)
    for _ in range(n_samples // 3):
        lats.append(np.random.uniform(36.0, 37.6))
        lons.append(np.random.uniform(8.5, 11.0))

    # Central region
    for _ in range(n_samples // 3):
        lats.append(np.random.uniform(34.0, 36.0))
        lons.append(np.random.uniform(8.0, 10.5))

    # South region
    for _ in range(n_samples - len(lats)):
        lats.append(np.random.uniform(30.2, 34.0))
        lons.append(np.random.uniform(7.5, 10.0))

    return pd.DataFrame({"name": names, "lat": lats, "lon": lons})