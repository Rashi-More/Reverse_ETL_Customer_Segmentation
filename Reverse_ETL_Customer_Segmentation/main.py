"""
Customer Segmentation Pipeline: DuckDB + Scikit-Learn K-Means + Reverse ETL to Simulated CRM.

Demonstration pipeline:
1. Ingests raw Kaggle customer data into DuckDB.
2. Validates data integrity (null checks, range validation, unique keys).
3. Applies K-Means clustering (k=5) on Annual Income and Spending Score.
4. Profiles clusters dynamically to assign business segment names and actionable recommendations.
5. Saves enriched segments back to DuckDB and exports to `crm_customers.csv` (Reverse ETL).
6. Generates a visual plot `customer_segments.png`.
"""

import csv
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "customers.csv"
DB_FILE = BASE_DIR / "customer_analytics.duckdb"
OUTPUT_FILE = BASE_DIR / "crm_customers.csv"
PLOT_FILE = BASE_DIR / "customer_segments.png"


def load_and_validate_data(con: duckdb.DuckDBPyConnection) -> int:
    """Loads raw CSV into DuckDB and runs validation checks."""
    input_path = str(INPUT_FILE).replace("\\", "/")

    # Step 1: Ingest into raw table
    con.execute(
        """
        CREATE OR REPLACE TABLE raw_customers AS
        SELECT
            CAST(CustomerID AS INTEGER) AS customer_id,
            TRIM(CAST(Genre AS VARCHAR)) AS gender,
            CAST(Age AS INTEGER) AS age,
            CAST("Annual_Income_(k$)" AS DOUBLE) AS annual_income,
            CAST(Spending_Score AS INTEGER) AS spending_score
        FROM read_csv_auto(?, header = true)
        """,
        [input_path],
    )

    # Step 2: Validate data integrity
    total_records = con.execute(
        "SELECT COUNT(*) FROM raw_customers"
    ).fetchone()[0]

    if total_records == 0:
        raise ValueError("The customer dataset is empty.")

    # Null or invalid check
    invalid_records = con.execute(
        """
        SELECT COUNT(*)
        FROM raw_customers
        WHERE customer_id IS NULL
           OR age IS NULL
           OR annual_income IS NULL
           OR spending_score IS NULL
           OR customer_id <= 0
           OR age <= 0
           OR annual_income < 0
           OR spending_score < 1
           OR spending_score > 100
        """
    ).fetchone()[0]

    if invalid_records > 0:
        raise ValueError(f"Data validation failed: Found {invalid_records} invalid record(s).")

    # Duplicate check
    duplicate_records = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT customer_id
            FROM raw_customers
            GROUP BY customer_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_records > 0:
        raise ValueError(f"Data validation failed: Found {duplicate_records} duplicate customer ID(s).")

    return total_records


def profile_segment(income: float, spending: float) -> tuple[str, str]:
    """
    Dynamically maps cluster centroids to understandable business segments
    and recommended CRM actions.
    """
    if income > 60 and spending > 60:
        return (
            "High Income, High Spending",
            "Potential premium customers: Offer VIP loyalty benefits, exclusive previews, and concierge service.",
        )
    elif income > 60 and spending < 40:
        return (
            "High Income, Low Spending",
            "Potential to increase engagement: Target with high-value offers, personalized campaigns, and incentives.",
        )
    elif income < 40 and spending > 60:
        return (
            "Low Income, High Spending",
            "High spending despite lower income: Target with trend-focused flash sales, rewards, and financing plans.",
        )
    elif income < 40 and spending < 40:
        return (
            "Low Income, Low Spending",
            "Price-conscious customers: Provide budget-friendly offers, essential bundles, and discount coupons.",
        )
    else:
        return (
            "Moderate Income, Moderate Spending",
            "Middle-income / average-spending baseline: Retain with standard seasonal promotions and loyalty perks.",
        )


def run_clustering_and_enrichment(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Fetches customer records from DuckDB, applies K-Means, and enriches records."""
    # Pull data into pandas for ML modeling
    df = con.execute(
        """
        SELECT customer_id, gender, age, annual_income, spending_score
        FROM raw_customers
        ORDER BY customer_id
        """
    ).df()

    features = df[["annual_income", "spending_score"]]

    # K-Means with k=5
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(features)

    # Profile centroids to assign human-readable labels and recommendations
    centroids = kmeans.cluster_centers_
    cluster_profiles = {}
    for cluster_id, (centroid_income, centroid_spending) in enumerate(centroids):
        name, action = profile_segment(centroid_income, centroid_spending)
        cluster_profiles[cluster_id] = {
            "name": name,
            "action": action,
            "centroid_income": round(centroid_income, 1),
            "centroid_spending": round(centroid_spending, 1),
        }

    df["segment_name"] = df["cluster"].map(lambda c: cluster_profiles[c]["name"])
    df["recommended_action"] = df["cluster"].map(lambda c: cluster_profiles[c]["action"])

    # Store back into DuckDB as analytical table
    con.register("df_enriched_view", df)
    con.execute(
        """
        CREATE OR REPLACE TABLE customer_segments AS
        SELECT
            customer_id,
            gender,
            age,
            annual_income,
            spending_score,
            cluster,
            segment_name,
            recommended_action
        FROM df_enriched_view
        """
    )
    con.unregister("df_enriched_view")

    return df


def export_reverse_etl(con: duckdb.DuckDBPyConnection) -> int:
    """Exports transformed segments to the simulated CRM CSV."""
    crm_query = """
        SELECT
            customer_id AS CustomerID,
            age AS Age,
            annual_income AS Annual_Income,
            spending_score AS Spending_Score,
            cluster AS Cluster,
            segment_name AS Segment_Name,
            recommended_action AS Recommended_Action
        FROM customer_segments
        ORDER BY CustomerID
    """

    result = con.execute(crm_query)
    columns = [col[0] for col in result.description]
    rows = result.fetchall()

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(rows)

    return len(rows)


def plot_segments(df: pd.DataFrame) -> None:
    """Creates and saves a scatter plot of customer segments with centroids."""
    plt.figure(figsize=(10, 6), dpi=150)

    # Consistent color palette for the 5 segments
    palette = {
        "High Income, High Spending": "#2a9d8f",
        "High Income, Low Spending": "#e76f51",
        "Low Income, High Spending": "#e9c46a",
        "Low Income, Low Spending": "#9b5de5",
        "Moderate Income, Moderate Spending": "#457b9d",
    }

    for segment, color in palette.items():
        subset = df[df["segment_name"] == segment]
        if not subset.empty:
            plt.scatter(
                subset["annual_income"],
                subset["spending_score"],
                s=60,
                color=color,
                alpha=0.85,
                edgecolors="none",
                label=f"{segment} (n={len(subset)})",
            )

    # Plot Cluster Centroids
    centroids = df.groupby("segment_name")[["annual_income", "spending_score"]].mean()
    plt.scatter(
        centroids["annual_income"],
        centroids["spending_score"],
        s=160,
        color="#1d3557",
        edgecolors="white",
        linewidths=2,
        marker="X",
        label="Cluster Centroids",
        zorder=5,
    )

    plt.title("Customer Segmentation via K-Means Clustering", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Annual Income (k$)", fontsize=11, fontweight="medium")
    plt.ylabel("Spending Score (1-100)", fontsize=11, fontweight="medium")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)
    plt.tight_layout()

    plt.savefig(PLOT_FILE)
    plt.close()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found at {INPUT_FILE}")

    print("=" * 65)
    print("DEMO PIPELINE: K-MEANS CUSTOMER SEGMENTATION + REVERSE ETL")
    print("=" * 65)

    con = duckdb.connect(str(DB_FILE))
    try:
        # Step 1 & 2: Load and validate in DuckDB
        print("\n[Step 1 & 2] Loading and validating records in DuckDB...")
        record_count = load_and_validate_data(con)
        print(f" -> Validation Passed: {record_count} valid records stored in table 'raw_customers'.")

        # Step 3 & 4: K-Means Clustering and Enrichment
        print("\n[Step 3 & 4] Applying K-Means clustering (k=5) and enriching segments...")
        df_segmented = run_clustering_and_enrichment(con)
        print(" -> Clustering complete and persisted in table 'customer_segments'.")

        # Display Segment Summary from DuckDB
        print("\nCUSTOMER SEGMENT DISTRIBUTION & STATS:")
        print("-" * 65)
        summary = con.execute(
            """
            SELECT
                segment_name,
                COUNT(*) AS total_customers,
                ROUND(AVG(annual_income), 1) AS avg_income,
                ROUND(AVG(spending_score), 1) AS avg_spending
            FROM customer_segments
            GROUP BY segment_name
            ORDER BY total_customers DESC
            """
        ).fetchall()

        for seg, cnt, inc, sp in summary:
            print(f" - {seg:<35} : {cnt:>2} customers | Avg Income: ${inc:>4}k | Avg Spending: {sp:>4.1f}")

        # Step 5: Reverse ETL to simulated CRM
        print("\n[Step 5] Performing Reverse ETL export to CRM...")
        exported_count = export_reverse_etl(con)
        print(f" -> Successfully exported {exported_count} records to: {OUTPUT_FILE.name}")

        # Visualization
        print("\n[Visualization] Generating cluster scatter plot...")
        plot_segments(df_segmented)
        print(f" -> Cluster plot saved to: {PLOT_FILE.name}")

        # CRM Export Sample Preview
        print("\nCRM EXPORT PREVIEW (Top 5 records):")
        print("-" * 65)
        preview_rows = con.execute(
            """
            SELECT CustomerID, Age, Annual_Income, Spending_Score, Segment_Name, Recommended_Action
            FROM read_csv_auto(?)
            LIMIT 5
            """,
            [str(OUTPUT_FILE).replace("\\", "/")],
        ).fetchall()

        for r in preview_rows:
            print(f"ID {r[0]:>3} | Age {r[1]:>2} | Inc ${r[2]:>3}k | Score {r[3]:>2} | {r[4]}")
            print(f"   -> Action: {r[5]}")

        print("=" * 65)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 65)

    finally:
        con.close()


if __name__ == "__main__":
    main()
