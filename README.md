# TerraGuard AI 🌍

### AI-Based Early Warning & Landslide Risk Monitoring System for the North Eastern Region

TerraGuard AI is an AI-powered landslide risk monitoring and early-warning prototype designed to identify and visualize potentially vulnerable regions using environmental, terrain, satellite-derived and historical landslide information.

The system combines GIS visualization, environmental data analysis and a Random Forest-based susceptibility inference pipeline to provide location-specific risk insights and what-if rainfall simulation.

---

## 🚨 Problem

The North Eastern Region of India is highly vulnerable to landslides due to:

- Intense and prolonged rainfall
- Steep and unstable terrain
- Soil moisture variation
- Complex elevation and slope conditions
- Historical landslide activity
- Changing environmental conditions

Traditional monitoring systems can make it difficult to combine these factors into a single, understandable risk view.

---

## 💡 Our Solution

TerraGuard AI combines multiple environmental and geospatial factors into an interactive risk-monitoring dashboard.

### Core Pipeline

Rainfall + Soil Moisture + Slope + Elevation + Historical Landslides + Geospatial Data

↓

Feature Processing

↓

Random Forest Susceptibility Model

↓

Risk Inference

↓

GIS Visualization + Alerts + What-if Simulation

---

## 🧠 Key Features

- 🗺️ Interactive GIS-based risk map
- 🌧️ Rainfall and environmental monitoring
- ⛰️ DEM-derived elevation and slope analysis
- 🛰️ Satellite-derived land-use and vegetation information
- 🤖 Random Forest ML susceptibility inference
- 📊 Explainable risk analysis
- ⚠️ Early-warning risk indicators
- 🎚️ Rainfall what-if simulation
- 📍 Location-specific risk assessment
- 📱 Disaster-management dashboard

---

## 🤖 Machine Learning

The prototype uses a **Random Forest classifier** for landslide susceptibility inference.

The model uses environmental and geospatial conditioning factors including:

- Curvature
- Slope
- Aspect
- Elevation
- NDVI
- Precipitation
- Land Use / Land Cover

### Prototype Evaluation

- Balanced Accuracy: **0.75**
- Macro F1 Score: **0.828**

The underlying susceptibility dataset contains samples from **Raigad district, Maharashtra**.

> **Important:** The current model is a prototype inference pipeline and has not yet been regionally validated for the North Eastern Region. Regional training and validation would be required before real-world deployment.

---

## 🌐 Data Sources

TerraGuard AI uses or demonstrates integration with:

- Smart India Hackathon 2026 — SIH26001
- Copernicus DEM
- Google Earth Engine / Dynamic World
- Open-Meteo weather API
- Mendeley landslide susceptibility dataset
- OpenStreetMap
- Leaflet.js

---

## 🏗️ System Architecture

```text
                    TERRAGUARD AI
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     Weather           Terrain          Satellite
      Data               DEM              Data
        │                │                │
        └────────────────┼────────────────┘
                         │
                  Feature Processing
                         │
                         ▼
                Random Forest Model
                         │
                         ▼
                  Risk Inference
                         │
              ┌──────────┴──────────┐
              │                     │
          Risk Score            Risk Level
              │                     │
              └──────────┬──────────┘
                         ▼
                  GIS Dashboard
                         │
              ┌──────────┴──────────┐
              │                     │
           Alerts              Simulation
