import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from data import fetch_tunisian_municipalities
from model import solve_airbase_problem_gurobi


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt5."""

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class AirbaseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results = None
        self.df_cities = None
        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('Airbase Location Optimization - Tunisia')
        self.setGeometry(100, 100, 1400, 900)

        # Apply stylesheet for better appearance
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QTabWidget::pane {
                border: 1px solid #c4c4c4;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e8eaed;
                padding: 10px;
                margin-right: 2px;
                border: 1px solid #c4c4c4;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2196F3;
            }
            QLabel {
                padding: 2px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #e0e0e0;
                selection-background-color: #bbdefb;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_control_tab()
        self.create_visualization_tab()
        self.create_distance_table_tab()
        self.create_city_data_tab()
        self.create_base_costs_tab()  # NEW TAB

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready to start optimization')

        # Load data
        self.load_city_data()

    def create_control_tab(self):
        """Create the control tab with parameters and buttons."""
        control_tab = QWidget()
        layout = QVBoxLayout(control_tab)

        # Title
        title_label = QLabel('Airbase Location Optimization - Tunisia')
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3; padding: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel('This application optimizes airbase locations in Tunisia using mathematical programming.')
        desc_label.setStyleSheet("font-size: 14px; color: #666; padding: 10px;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Parameters group
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QGridLayout()

        # Row 0: Basic parameters
        params_layout.addWidget(QLabel("Coverage Radius (km):"), 0, 0)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(50, 500)
        self.radius_spin.setValue(150)
        self.radius_spin.setSuffix(" km")
        params_layout.addWidget(self.radius_spin, 0, 1)

        params_layout.addWidget(QLabel("Budget:"), 0, 2)
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(50, 500)
        self.budget_spin.setValue(120)
        params_layout.addWidget(self.budget_spin, 0, 3)

        # Row 1: Capacity and zones
        params_layout.addWidget(QLabel("Min Capacity Required:"), 1, 0)
        self.capacity_spin = QSpinBox()
        self.capacity_spin.setRange(20, 200)
        self.capacity_spin.setValue(40)
        params_layout.addWidget(self.capacity_spin, 1, 1)

        params_layout.addWidget(QLabel("Number of Zones:"), 1, 2)
        self.zones_spin = QSpinBox()
        self.zones_spin.setRange(5, 100)
        self.zones_spin.setValue(25)
        params_layout.addWidget(self.zones_spin, 1, 3)

        # Row 2: Sites and min bases
        params_layout.addWidget(QLabel("Number of Sites:"), 2, 0)
        self.sites_spin = QSpinBox()
        self.sites_spin.setRange(5, 50)
        self.sites_spin.setValue(12)
        params_layout.addWidget(self.sites_spin, 2, 1)

        params_layout.addWidget(QLabel("Min Bases Required:"), 2, 2)
        self.min_bases_spin = QSpinBox()
        self.min_bases_spin.setRange(1, 10)
        self.min_bases_spin.setValue(2)
        params_layout.addWidget(self.min_bases_spin, 2, 3)

        # Row 3: Weight parameters
        params_layout.addWidget(QLabel("Weight  (Bases):"), 3, 0)
        self.lambda1_spin = QDoubleSpinBox()
        self.lambda1_spin.setRange(0.0, 10.0)
        self.lambda1_spin.setValue(0.3)
        self.lambda1_spin.setSingleStep(0.1)
        params_layout.addWidget(self.lambda1_spin, 3, 1)

        params_layout.addWidget(QLabel("Weight  (Cost):"), 3, 2)
        self.lambda2_spin = QDoubleSpinBox()
        self.lambda2_spin.setRange(0.0, 10.0)
        self.lambda2_spin.setValue(0.5)
        self.lambda2_spin.setSingleStep(0.1)
        params_layout.addWidget(self.lambda2_spin, 3, 3)

        # Tooltips for weights
        self.lambda1_spin.setToolTip("Lower value encourages more bases")
        self.lambda2_spin.setToolTip("Higher value emphasizes cost savings")

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Data info group
        data_group = QGroupBox("Data Information")
        data_layout = QVBoxLayout()

        self.data_info_label = QLabel("No data loaded yet.")
        self.data_info_label.setStyleSheet("font-size: 12px; padding: 10px;")
        data_layout.addWidget(self.data_info_label)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Button layout
        button_layout = QHBoxLayout()

        # Run optimization button
        self.run_button = QPushButton("Run Optimization")
        self.run_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.clicked.connect(self.run_optimization)
        self.run_button.setMinimumHeight(50)
        button_layout.addWidget(self.run_button)

        # Reload data button
        self.reload_button = QPushButton("Reload City Data")
        self.reload_button.setIcon(QApplication.style().standardIcon(QStyle.SP_BrowserReload))
        self.reload_button.clicked.connect(self.load_city_data)
        button_layout.addWidget(self.reload_button)

        layout.addLayout(button_layout)

        # Results summary
        self.results_label = QLabel("")
        self.results_label.setStyleSheet(
            "font-size: 14px; padding: 15px; background-color: #e8f4fd; border-radius: 5px;")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

        layout.addStretch()
        self.tab_widget.addTab(control_tab, "Control Panel")

    def create_visualization_tab(self):
        """Create the visualization tab with matplotlib canvas."""
        viz_tab = QWidget()
        layout = QVBoxLayout(viz_tab)

        # Title
        title_label = QLabel('Geographical Visualization')
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Matplotlib canvas
        self.canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.canvas.axes.set_facecolor('#f8f9fa')

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Info label
        self.viz_info_label = QLabel("Run optimization to see visualization")
        self.viz_info_label.setStyleSheet("font-size: 12px; color: #666; padding: 10px;")
        self.viz_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.viz_info_label)

        self.tab_widget.addTab(viz_tab, "Visualization")

    def create_distance_table_tab(self):
        """Create the tab with distance table."""
        table_tab = QWidget()
        layout = QVBoxLayout(table_tab)

        # Title
        title_label = QLabel('Distance Matrix: Zones to Potential Sites')
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Filter layout
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show distances for:"))

        self.distance_filter_combo = QComboBox()
        self.distance_filter_combo.addItems(["All Sites", "Selected Sites Only", "Unselected Sites Only"])
        filter_layout.addWidget(self.distance_filter_combo)

        self.distance_filter_combo.currentTextChanged.connect(self.update_distance_table)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Table widget
        self.distance_table = QTableWidget()
        self.distance_table.setAlternatingRowColors(True)
        self.distance_table.setSortingEnabled(True)
        layout.addWidget(self.distance_table)

        self.tab_widget.addTab(table_tab, "Distance Matrix")

    def create_city_data_tab(self):
        """Create the tab with city data table."""
        city_tab = QWidget()
        layout = QVBoxLayout(city_tab)

        # Title
        title_label = QLabel('City Data')
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # City table
        self.city_table = QTableWidget()
        self.city_table.setAlternatingRowColors(True)
        self.city_table.setSortingEnabled(True)
        layout.addWidget(self.city_table)

        self.tab_widget.addTab(city_tab, "City Data")

    def create_base_costs_tab(self):
        """Create the tab with base costs table."""
        base_costs_tab = QWidget()
        layout = QVBoxLayout(base_costs_tab)

        # Title
        title_label = QLabel('Base Costs and Characteristics')
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel('All potential bases with their randomly generated costs, risks, and capacities')
        desc_label.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Filter layout
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by:"))

        self.base_filter_combo = QComboBox()
        self.base_filter_combo.addItems(["All Bases", "Selected Bases Only", "Unselected Bases Only"])
        filter_layout.addWidget(self.base_filter_combo)

        self.base_filter_combo.currentTextChanged.connect(self.update_base_costs_table)

        filter_layout.addStretch()

        # Sort layout
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort by:"))

        self.base_sort_combo = QComboBox()
        self.base_sort_combo.addItems(["Site Name", "Cost (Low to High)", "Cost (High to Low)",
                                       "Risk (Low to High)", "Risk (High to Low)",
                                       "Capacity (Low to High)", "Capacity (High to Low)"])
        self.base_sort_combo.currentTextChanged.connect(self.update_base_costs_table)
        sort_layout.addWidget(self.base_sort_combo)

        filter_layout.addLayout(sort_layout)
        layout.addLayout(filter_layout)

        # Table widget for base costs
        self.base_costs_table = QTableWidget()
        self.base_costs_table.setAlternatingRowColors(True)
        self.base_costs_table.setSortingEnabled(True)
        layout.addWidget(self.base_costs_table)

        # Summary label
        self.base_summary_label = QLabel("")
        self.base_summary_label.setStyleSheet(
            "font-size: 12px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        self.base_summary_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.base_summary_label)

        self.tab_widget.addTab(base_costs_tab, "Base Costs")

    def update_base_costs_table(self):
        """Update the base costs table with current results."""
        if not self.results or 'results_df' not in self.results:
            self.base_costs_table.clear()
            self.base_costs_table.setRowCount(0)
            self.base_costs_table.setColumnCount(0)
            self.base_summary_label.setText("Run optimization to see base costs")
            return

        # Get the results DataFrame
        results_df = self.results['results_df'].copy()
        selected_sites = self.results['selected_sites']
        budget = self.results['parameters']['B']
        total_cost = self.results['metrics']['total_cost']

        # Filter based on selection
        filter_text = self.base_filter_combo.currentText()
        if filter_text == "Selected Bases Only":
            results_df = results_df[results_df['Opened'] == 1]
        elif filter_text == "Unselected Bases Only":
            results_df = results_df[results_df['Opened'] == 0]

        # Sort based on selection
        sort_text = self.base_sort_combo.currentText()
        if sort_text == "Cost (Low to High)":
            results_df = results_df.sort_values('Cost', ascending=True)
        elif sort_text == "Cost (High to Low)":
            results_df = results_df.sort_values('Cost', ascending=False)
        elif sort_text == "Risk (Low to High)":
            results_df = results_df.sort_values('Risk', ascending=True)
        elif sort_text == "Risk (High to Low)":
            results_df = results_df.sort_values('Risk', ascending=False)
        elif sort_text == "Capacity (Low to High)":
            results_df = results_df.sort_values('Capacity', ascending=True)
        elif sort_text == "Capacity (High to Low)":
            results_df = results_df.sort_values('Capacity', ascending=False)
        elif sort_text == "Site Name":
            results_df = results_df.sort_values('Site')

        # Update table
        self.base_costs_table.setRowCount(len(results_df))
        self.base_costs_table.setColumnCount(7)
        self.base_costs_table.setHorizontalHeaderLabels(['Site', 'Selected', 'Cost', 'Risk', 'Capacity',
                                                         'Latitude', 'Longitude'])

        for i, row in results_df.iterrows():
            # Site name
            site_item = QTableWidgetItem(str(row['Site']))
            self.base_costs_table.setItem(i, 0, site_item)

            # Selected status
            selected_item = QTableWidgetItem("Yes" if row['Opened'] == 1 else "No")
            selected_item.setForeground(QColor(0, 100, 0) if row['Opened'] == 1 else QColor(150, 0, 0))
            selected_item.setFont(QFont("Arial", 9, QFont.Bold) if row['Opened'] == 1 else QFont("Arial", 9))
            self.base_costs_table.setItem(i, 1, selected_item)

            # Cost
            cost_item = QTableWidgetItem(f"{row['Cost']}")
            cost_item.setForeground(QColor(200, 0, 0))
            cost_item.setFont(QFont("Arial", 9, QFont.Bold))
            self.base_costs_table.setItem(i, 2, cost_item)

            # Risk (1-10 scale)
            risk_item = QTableWidgetItem(f"{row['Risk']}")
            # Color code risk: green for low risk, yellow for medium, red for high
            risk_value = row['Risk']
            if risk_value <= 3:
                risk_item.setForeground(QColor(0, 150, 0))
            elif risk_value <= 7:
                risk_item.setForeground(QColor(200, 150, 0))
            else:
                risk_item.setForeground(QColor(200, 0, 0))
            self.base_costs_table.setItem(i, 3, risk_item)

            # Capacity
            capacity_item = QTableWidgetItem(f"{row['Capacity']}")
            capacity_item.setForeground(QColor(0, 0, 150))
            self.base_costs_table.setItem(i, 4, capacity_item)

            # Latitude
            lat_item = QTableWidgetItem(f"{row['Latitude']:.4f}")
            self.base_costs_table.setItem(i, 5, lat_item)

            # Longitude
            lon_item = QTableWidgetItem(f"{row['Longitude']:.4f}")
            self.base_costs_table.setItem(i, 6, lon_item)

            # Color the entire row based on selection
            if row['Opened'] == 1:
                for col in range(7):
                    item = self.base_costs_table.item(i, col)
                    if item:
                        item.setBackground(QColor(220, 255, 220))  # Light green for selected bases

        # Adjust column widths
        self.base_costs_table.resizeColumnsToContents()
        self.base_costs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        # Update summary label
        selected_count = len(results_df[results_df['Opened'] == 1])
        total_count = len(results_df)
        avg_cost = results_df['Cost'].mean() if len(results_df) > 0 else 0
        avg_risk = results_df['Risk'].mean() if len(results_df) > 0 else 0
        avg_capacity = results_df['Capacity'].mean() if len(results_df) > 0 else 0

        summary_text = f"Showing {total_count} bases ({selected_count} selected). "
        summary_text += f"Average Cost: {avg_cost:.1f}, Average Risk: {avg_risk:.1f}, Average Capacity: {avg_capacity:.1f}. "
        summary_text += f"Total selected cost: {total_cost:.1f} / {budget} ({total_cost / budget * 100:.1f}% of budget)"

        self.base_summary_label.setText(summary_text)

    def load_city_data(self):
        """Load city data from API."""
        self.status_bar.showMessage('Loading city data...')
        QApplication.processEvents()

        try:
            self.df_cities = fetch_tunisian_municipalities()

            # Update data info label
            cities_count = len(self.df_cities)
            lat_range = f"{self.df_cities['lat'].min():.2f} to {self.df_cities['lat'].max():.2f}"
            lon_range = f"{self.df_cities['lon'].min():.2f} to {self.df_cities['lon'].max():.2f}"

            self.data_info_label.setText(
                f"Loaded {cities_count} municipalities\n"
                f"Latitude range: {lat_range}\n"
                f"Longitude range: {lon_range}"
            )

            # Populate city data table
            self.update_city_table()

            self.status_bar.showMessage(f'Successfully loaded {cities_count} municipalities', 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load city data: {str(e)}")
            self.status_bar.showMessage('Error loading city data', 5000)

    def update_city_table(self):
        """Update the city data table."""
        if self.df_cities is not None:
            self.city_table.setRowCount(len(self.df_cities))
            self.city_table.setColumnCount(3)
            self.city_table.setHorizontalHeaderLabels(['City Name', 'Latitude', 'Longitude'])

            for i, row in self.df_cities.iterrows():
                # City name (string)
                self.city_table.setItem(i, 0, QTableWidgetItem(str(row['name'])))

                # Latitude (float)
                try:
                    lat_value = float(row['lat'])
                    self.city_table.setItem(i, 1, QTableWidgetItem(f"{lat_value:.4f}"))
                except (ValueError, TypeError):
                    self.city_table.setItem(i, 1, QTableWidgetItem(str(row['lat'])))

                # Longitude (float)
                try:
                    lon_value = float(row['lon'])
                    self.city_table.setItem(i, 2, QTableWidgetItem(f"{lon_value:.4f}"))
                except (ValueError, TypeError):
                    self.city_table.setItem(i, 2, QTableWidgetItem(str(row['lon'])))

            self.city_table.resizeColumnsToContents()

    def run_optimization(self):
        """Run the optimization with current parameters."""
        if self.df_cities is None:
            QMessageBox.warning(self, "Warning", "Please load city data first!")
            return

        self.status_bar.showMessage('Running optimization...')
        self.run_button.setEnabled(False)
        QApplication.processEvents()

        try:
            # Get parameters from UI
            R = self.radius_spin.value()
            B = self.budget_spin.value()
            P_min = self.capacity_spin.value()
            lambda1 = self.lambda1_spin.value()
            lambda2 = self.lambda2_spin.value()
            n_zones = self.zones_spin.value()
            n_sites = self.sites_spin.value()
            k_min = self.min_bases_spin.value()

            # Run optimization with fixed parameters
            self.results = solve_airbase_problem_gurobi(
                df_cities=self.df_cities,
                R=R,
                B=B,
                P_min=P_min,
                lambda1=lambda1,
                lambda2=lambda2,
                n_zones=n_zones,
                n_sites=n_sites,
                k_min=k_min
            )

            if self.results:
                # Update visualization
                self.update_visualization()

                # Update distance table
                self.update_distance_table()

                # Update base costs table
                self.update_base_costs_table()

                # Update results label
                metrics = self.results['metrics']
                params = self.results['parameters']

                summary = f"""
                Optimization Complete!
                Selected Bases: {metrics['num_bases']} (min required: {params['k_min']})
                Total Cost: {metrics['total_cost']:.1f} / {params['B']} ({metrics['total_cost'] / params['B'] * 100:.1f}%)
                Total Risk: {metrics['total_risk']:.1f}
                Total Capacity: {metrics['total_capacity']:.1f} (min required: {params['P_min']})
                Objective Value: {metrics['objective']:.2f}
                """
                self.results_label.setText(summary)

                self.status_bar.showMessage('Optimization completed successfully!', 5000)

                # Switch to visualization tab
                self.tab_widget.setCurrentIndex(1)

            else:
                QMessageBox.warning(self, "Warning",
                                    "Optimization failed to find a solution. Try adjusting parameters.")
                self.status_bar.showMessage('Optimization failed', 5000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Optimization error: {str(e)}")
            self.status_bar.showMessage('Error during optimization', 5000)

        finally:
            self.run_button.setEnabled(True)

    def update_visualization(self):
        """Update the matplotlib visualization."""
        if not self.results:
            return

        # Clear previous plot
        self.canvas.axes.clear()

        zones = self.results["zones"]
        sites = self.results["sites"]
        coords_zones = self.results["coords_zones"]
        coords_sites = self.results["coords_sites"]
        x_vars = self.results["x_vars"]
        params = self.results["parameters"]
        R = params["R"]

        # Convert coordinates to numpy arrays
        zones_lons = coords_zones[:, 1]
        zones_lats = coords_zones[:, 0]
        sites_lons = coords_sites[:, 1]
        sites_lats = coords_sites[:, 0]

        # Plot zones
        zone_colors = ['blue'] * len(zones)
        self.canvas.axes.scatter(zones_lons, zones_lats,
                                 marker='o', s=80, alpha=0.7, label="Zones",
                                 c=zone_colors, edgecolors='black', linewidth=0.5, zorder=3)

        # Plot potential sites
        self.canvas.axes.scatter(sites_lons, sites_lats,
                                 s=120, alpha=0.6, label="Potential sites",
                                 c='gray', marker='^', edgecolors='black', linewidth=0.5, zorder=4)

        # Plot selected bases
        selected_count = 0
        selected_coords = []

        for j_idx, j in enumerate(sites):
            if x_vars[j].X > 0.5:
                selected_count += 1
                lat, lon = sites_lats[j_idx], sites_lons[j_idx]
                selected_coords.append((lon, lat, j_idx, j))

        # Sort selected bases by latitude for consistent coloring
        selected_coords.sort(key=lambda x: x[1])  # Sort by latitude

        base_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF']

        for idx, (lon, lat, j_idx, j) in enumerate(selected_coords):
            base_color = base_colors[idx % len(base_colors)]

            # Draw selected base
            self.canvas.axes.scatter(lon, lat, s=350, edgecolor='black', linewidth=2.5,
                                     marker='s', facecolor=base_color,
                                     label=f"Base {idx + 1}" if idx == 0 else "",
                                     zorder=10)

            # Add label
            self.canvas.axes.annotate(f"B{idx + 1}", (lon, lat),
                                      xytext=(8, 8), textcoords='offset points',
                                      fontsize=9, fontweight='bold', color='black',
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor=base_color, alpha=0.7),
                                      zorder=11)

            # Draw coverage area
            radius_km = R
            radius_lat = radius_km / 111
            radius_lon = radius_km / (111 * np.cos(np.radians(lat)))

            ellipse = Ellipse((lon, lat),
                              width=2 * radius_lon,
                              height=2 * radius_lat,
                              fill=True, alpha=0.08,
                              linewidth=1.2, edgecolor=base_color,
                              facecolor=base_color, zorder=2)
            self.canvas.axes.add_patch(ellipse)

        # Set axis limits to show all of Tunisia
        tunisia_lon_min, tunisia_lon_max = 7.5, 11.6
        tunisia_lat_min, tunisia_lat_max = 30.2, 37.6

        # Add some margin
        lon_margin = (tunisia_lon_max - tunisia_lon_min) * 0.1
        lat_margin = (tunisia_lat_max - tunisia_lat_min) * 0.1

        self.canvas.axes.set_xlim(tunisia_lon_min - lon_margin, tunisia_lon_max + lon_margin)
        self.canvas.axes.set_ylim(tunisia_lat_min - lat_margin, tunisia_lat_max + lat_margin)

        # Customize plot
        self.canvas.axes.set_xlabel("Longitude", fontsize=11)
        self.canvas.axes.set_ylabel("Latitude", fontsize=11)
        self.canvas.axes.set_title("Airbase Locations and Coverage Areas", fontsize=13, fontweight='bold')
        self.canvas.axes.grid(True, alpha=0.2, linestyle='--')

        # Add scale bar (approx 100 km)
        mean_lat = np.mean([tunisia_lat_min, tunisia_lat_max])
        scale_bar_km = 100
        scale_bar_deg = scale_bar_km / (111 * np.cos(np.radians(mean_lat)))

        scale_x = tunisia_lon_min + lon_margin * 0.8
        scale_y = tunisia_lat_min + lat_margin * 0.8

        self.canvas.axes.plot([scale_x, scale_x + scale_bar_deg], [scale_y, scale_y],
                              'k-', linewidth=2.5, zorder=5)
        self.canvas.axes.text(scale_x + scale_bar_deg / 2, scale_y - lat_margin * 0.08,
                              f'{scale_bar_km} km', ha='center', fontsize=8, fontweight='bold')

        # Add legend
        handles, labels = self.canvas.axes.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            self.canvas.axes.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

        # Update info label
        self.viz_info_label.setText(
            f"Showing {len(zones)} zones and {len(sites)} potential sites. "
            f"{selected_count} bases selected with {R} km coverage radius."
        )

        # Refresh canvas
        self.canvas.draw()

    def update_distance_table(self):
        """Update the distance table with current results - FIXED FORMATTING."""
        if not self.results or 'distance_matrix' not in self.results:
            return

        distance_matrix = self.results['distance_matrix']
        selected_sites = self.results['selected_sites']
        sites = self.results['sites']
        R = self.results['parameters']['R']

        # Filter sites based on selection
        filter_text = self.distance_filter_combo.currentText()
        if filter_text == "Selected Sites Only":
            columns = [site for site in selected_sites if site in distance_matrix.columns]
        elif filter_text == "Unselected Sites Only":
            columns = [site for site in sites if site not in selected_sites and site in distance_matrix.columns]
        else:  # "All Sites"
            columns = sites

        # Create filtered dataframe
        filtered_df = distance_matrix[columns].copy()

        # Add minimum distance column - ensure numeric
        if len(columns) > 0:
            filtered_df['Min Distance (km)'] = filtered_df.min(axis=1)
            filtered_df['Nearest Site'] = filtered_df[columns].idxmin(axis=1)

        # Update table
        self.distance_table.setRowCount(len(filtered_df))
        self.distance_table.setColumnCount(len(filtered_df.columns) + 1)  # +1 for zone column

        # Set header labels
        header_labels = ['Zone'] + [str(col) for col in filtered_df.columns]
        self.distance_table.setHorizontalHeaderLabels(header_labels)

        # Fill table with data - FIXED FORMATTING ISSUE
        for i, (zone, row) in enumerate(filtered_df.iterrows()):
            # Zone name in first column
            self.distance_table.setItem(i, 0, QTableWidgetItem(str(zone)))

            # Distance values
            for j, col in enumerate(filtered_df.columns, start=1):
                value = row[col]

                # Create item based on column type
                if col == 'Nearest Site':
                    # This is a string column
                    item_text = str(value)
                    item = QTableWidgetItem(item_text)
                    item.setForeground(QColor(0, 0, 150))

                elif col == 'Min Distance (km)':
                    # This is a numeric column
                    try:
                        # Ensure value is numeric
                        if pd.isna(value):
                            num_value = float('nan')
                        else:
                            num_value = float(value)

                        if pd.isna(num_value):
                            item_text = "N/A"
                        else:
                            item_text = f"{num_value:.2f} km"

                        item = QTableWidgetItem(item_text)
                        item.setForeground(QColor(0, 100, 0))
                        item.setFont(QFont("Arial", 9, QFont.Bold))

                        # Color code based on distance
                        if not pd.isna(num_value) and num_value <= R:
                            item.setBackground(QColor(220, 255, 220))  # Light green for within range
                        elif not pd.isna(num_value):
                            item.setBackground(QColor(255, 220, 220))  # Light red for out of range

                    except (ValueError, TypeError):
                        item = QTableWidgetItem(str(value))

                else:
                    # Regular distance column (should be numeric)
                    try:
                        # Ensure value is numeric
                        if pd.isna(value):
                            num_value = float('nan')
                        else:
                            num_value = float(value)

                        if pd.isna(num_value):
                            item_text = "N/A"
                        else:
                            item_text = f"{num_value:.2f} km"

                        item = QTableWidgetItem(item_text)

                        # Color code based on distance
                        if not pd.isna(num_value) and num_value <= R:
                            item.setBackground(QColor(220, 255, 220))  # Light green for within range
                        elif not pd.isna(num_value):
                            item.setBackground(QColor(255, 220, 220))  # Light red for out of range

                    except (ValueError, TypeError):
                        # If conversion fails, just use string representation
                        item = QTableWidgetItem(str(value))

                self.distance_table.setItem(i, j, item)

        # Adjust column widths
        self.distance_table.resizeColumnsToContents()

        # Make the zone column stretch to fill available space
        if self.distance_table.columnCount() > 0:
            self.distance_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)