import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "car.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration for publication-grade figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EBEBEB'
plt.rcParams['grid.linestyle'] = '--'

def generate_eda_figures():
    print("Loading official dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Initial shape: {df.shape}")
    
    # Identify and drop exact duplicates
    dupes_count = df.duplicated().sum()
    print(f"Identified duplicate records: {dupes_count}")
    df_clean = df.drop_duplicates().copy()
    print(f"Cleaned shape: {df_clean.shape}")
    
    # Feature Engineering: Vehicle Age
    # Reference Year = 2018 (Dataset capture baseline)
    REF_YEAR = 2018
    df_clean['Vehicle_Age'] = REF_YEAR - df_clean['Year']
    
    # -------------------------------------------------------------
    # 1. Price Distribution (Selling Price vs Showroom Price)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    sns.histplot(df_clean['Selling_Price'], kde=True, color='#2563EB', ax=axes[0], bins=25, edgecolor='black', alpha=0.6)
    axes[0].axvline(df_clean['Selling_Price'].median(), color='#DC2626', linestyle='--', linewidth=2, label=f"Median: ₹{df_clean['Selling_Price'].median():.2f}L")
    axes[0].axvline(df_clean['Selling_Price'].mean(), color='#059669', linestyle=':', linewidth=2, label=f"Mean: ₹{df_clean['Selling_Price'].mean():.2f}L")
    axes[0].set_title("Distribution of Resale Selling Price (in Lakhs ₹)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Selling Price (Lakhs ₹)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(frameon=True, facecolor='white')
    
    sns.boxplot(x=df_clean['Selling_Price'], color='#93C5FD', ax=axes[1], flierprops=dict(marker='o', markerfacecolor='#DC2626', markersize=6))
    axes[1].set_title("Selling Price Boxplot & Outlier Detection", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Selling Price (Lakhs ₹)", fontsize=11)
    
    plt.tight_layout()
    p1 = FIG_DIR / "price_distribution.png"
    plt.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p1.name}")

    # -------------------------------------------------------------
    # 2. Fuel Type and Transmission Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    palette_fuel = {'Petrol': '#3B82F6', 'Diesel': '#10B981', 'CNG': '#F59E0B'}
    sns.barplot(data=df_clean, x='Fuel_Type', y='Selling_Price', hue='Fuel_Type', legend=False, estimator='mean', errorbar=None, palette=palette_fuel, ax=axes[0], edgecolor='black')
    for p in axes[0].patches:
        axes[0].annotate(f"₹{p.get_height():.2f}L", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    axes[0].set_title("Average Selling Price by Fuel Type", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Fuel Type", fontsize=11)
    axes[0].set_ylabel("Mean Selling Price (Lakhs ₹)", fontsize=11)
    
    palette_trans = {'Manual': '#6366F1', 'Automatic': '#EC4899'}
    sns.barplot(data=df_clean, x='Transmission', y='Selling_Price', hue='Transmission', legend=False, estimator='mean', errorbar=None, palette=palette_trans, ax=axes[1], edgecolor='black')
    for p in axes[1].patches:
        axes[1].annotate(f"₹{p.get_height():.2f}L", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    axes[1].set_title("Average Selling Price by Transmission Type", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Transmission", fontsize=11)
    axes[1].set_ylabel("Mean Selling Price (Lakhs ₹)", fontsize=11)
    
    plt.tight_layout()
    p2 = FIG_DIR / "fuel_transmission_distribution.png"
    plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p2.name}")

    # -------------------------------------------------------------
    # 3. Showroom Price (Present_Price) vs Selling Price
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    sns.scatterplot(data=df_clean, x='Present_Price', y='Selling_Price', hue='Transmission', palette={'Manual': '#2563EB', 'Automatic': '#DC2626'}, s=70, alpha=0.85, edgecolor='black', linewidth=0.5)
    sns.regplot(data=df_clean, x='Present_Price', y='Selling_Price', scatter=False, ax=plt.gca(), color='#4B5563', line_kws={'linestyle': '--', 'linewidth': 1.5, 'label': 'Linear Fit Trend'})
    plt.title("Resale Selling Price vs. Original Showroom Price", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Original Showroom Price (Present_Price in Lakhs ₹)", fontsize=12)
    plt.ylabel("Resale Selling Price (Lakhs ₹)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    
    p3 = FIG_DIR / "showroom_vs_selling.png"
    plt.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p3.name}")

    # -------------------------------------------------------------
    # 4. Vehicle Age (Years of Service) vs Selling Price
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5), dpi=300)
    sns.boxplot(data=df_clean, x='Vehicle_Age', y='Selling_Price', color='#93C5FD', linewidth=1)
    sns.stripplot(data=df_clean, x='Vehicle_Age', y='Selling_Price', color='#DC2626', size=5, jitter=0.2, alpha=0.6)
    plt.title("Depreciation Effect: Vehicle Age (Years of Service) vs. Selling Price", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Vehicle Age (Years from Baseline 2018)", fontsize=12)
    plt.ylabel("Selling Price (Lakhs ₹)", fontsize=12)
    
    p4 = FIG_DIR / "age_vs_selling.png"
    plt.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p4.name}")

    # -------------------------------------------------------------
    # 5. Kilometers Driven vs Selling Price
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    sns.scatterplot(data=df_clean, x='Kms_Driven', y='Selling_Price', hue='Fuel_Type', palette=palette_fuel, s=70, alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.title("Selling Price vs. Kilometers Driven", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Odometer Reading (Kilometers Driven)", fontsize=12)
    plt.ylabel("Selling Price (Lakhs ₹)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', title="Fuel Type")
    
    # Highlight 500,000 km outlier
    outlier = df_clean[df_clean['Kms_Driven'] > 400000]
    for _, row in outlier.iterrows():
        plt.annotate(f"High Mileage: {row['Kms_Driven']:,} km\n({row['Car_Name']}, ₹{row['Selling_Price']}L)",
                     xy=(row['Kms_Driven'], row['Selling_Price']),
                     xytext=(row['Kms_Driven'] - 140000, row['Selling_Price'] + 4),
                     arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
                     fontsize=9, fontweight='bold', color='#B91C1C',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEE2E2', edgecolor='#DC2626'))
    
    p5 = FIG_DIR / "kms_vs_selling.png"
    plt.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p5.name}")

    # -------------------------------------------------------------
    # 6. Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6.5), dpi=300)
    num_cols = ['Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner', 'Vehicle_Age']
    corr_mat = df_clean[num_cols].corr()
    
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                center=0, square=True, linewidths=1, cbar_kws={"shrink": .8}, annot_kws={"size": 11, "weight": "bold"})
    plt.title("Feature Correlation Matrix Heatmap", fontsize=14, fontweight='bold', pad=14)
    
    p6 = FIG_DIR / "correlation_heatmap.png"
    plt.savefig(p6, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p6.name}")

    # -------------------------------------------------------------
    # 7. Seller Type and Owner Analysis
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    palette_seller = {'Dealer': '#0284C7', 'Individual': '#F97316'}
    sns.boxplot(data=df_clean, x='Seller_Type', y='Selling_Price', hue='Seller_Type', legend=False, palette=palette_seller, ax=axes[0])
    axes[0].set_title("Selling Price Distribution by Seller Type", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Seller Type", fontsize=11)
    axes[0].set_ylabel("Selling Price (Lakhs ₹)", fontsize=11)
    
    sns.barplot(data=df_clean, x='Owner', y='Selling_Price', color='#8B5CF6', ax=axes[1], edgecolor='black')
    for p in axes[1].patches:
        axes[1].annotate(f"₹{p.get_height():.2f}L", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    axes[1].set_title("Mean Selling Price by Number of Previous Owners", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Previous Owners Count", fontsize=11)
    axes[1].set_ylabel("Mean Selling Price (Lakhs ₹)", fontsize=11)
    
    plt.tight_layout()
    p7 = FIG_DIR / "seller_owner_analysis.png"
    plt.savefig(p7, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p7.name}")

    print("\nAll 7 publication-grade EDA figures generated successfully!")

if __name__ == "__main__":
    generate_eda_figures()
