import json
from datetime import datetime
from typing import Any, Dict

import ipdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def read_conferences(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def create_conference_timeline(conferences_data: Dict[str, Any]):
    """Create a timeline DataFrame from conference data"""
    conferences = conferences_data.get("conferences", [])

    timeline_data = []
    for conf in conferences:
        start = conf.get("start", "")
        end = conf.get("end", "")
        name = conf.get("name", "")
        level = conf.get("level", "")

        # Parse dates
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        now_date = datetime.now()

        # Modify year for start_date (example: set to 2024)
        # You can change the year here
        target_year = now_date.year

        # Create timeline entry
        timeline_data.append(
            {
                "Conference": name,
                "Passed": now_date > end_date,
                "Start": start_date.replace(year=target_year),
                "End": end_date.replace(year=target_year),
                "Level": level,
                "URL": conf.get("url", ""),
            }
        )

    return pd.DataFrame(timeline_data)


def main():
    # Read conference data
    conferences_data = read_conferences("meta/search/conferences.json")

    # Create DataFrame
    df = create_conference_timeline(conferences_data)

    # sort df by deadline
    df = df.sort_values("End", ascending=False)

    print("Conference Timeline DataFrame:")
    print(df.head(10))
    print(f"\nTotal conferences: {len(df)}")

    # plot the df data as a bar chart
    plt.figure(figsize=(10, max(6, len(df) * 0.2)))

    # Create horizontal bar chart
    y_pos = range(len(df))

    # Custom soft and beautiful colors for all conferences
    soft_colors = [
        "#FFB6C1",  # Light pink
        "#98FB98",  # Pale green
        "#87CEEB",  # Sky blue
        "#DDA0DD",  # Plum
        "#F0E68C",  # Khaki
        "#E6E6FA",  # Lavender
        "#FFA07A",  # Light salmon
        "#B0E0E6",  # Powder blue
        "#F5DEB3",  # Wheat
        "#D8BFD8",  # Thistle
        "#FFE4B5",  # Moccasin
        "#E0FFFF",  # Light cyan
        "#F0FFF0",  # Honeydew
        "#FFF0F5",  # Lavender blush
        "#F5F5DC",  # Beige
    ]

    colors = []
    patterns = []
    color_index = 0
    for _, row in df.iterrows():
        colors.append(soft_colors[color_index % len(soft_colors)])
        if row["Passed"]:
            # Past conferences - use diagonal stripes pattern
            patterns.append("////")
        else:
            # Future conferences - use solid pattern
            patterns.append("")
        color_index += 1

    # Create bars with different patterns
    bars = plt.barh(y_pos, width=df["End"] - df["Start"], left=df["Start"], height=0.8, color=colors)

    # Apply patterns to bars
    for i, (bar, pattern) in enumerate(zip(bars, patterns)):
        if pattern:  # If pattern exists (for passed conferences)
            bar.set_hatch(pattern)

    # Set y-axis labels (conference names)
    plt.yticks(y_pos, df["Conference"])

    # Add current date vertical line
    current_date = datetime.now()
    plt.axvline(
        x=current_date,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f'Current Date: {current_date.strftime("%Y-%m-%d")}',
    )

    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Add labels and title
    plt.xlabel("Month")
    plt.ylabel("Conference Name")
    plt.title("Conference Timeline")
    plt.grid(True)

    # Create custom legend for conference status
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#FFB6C1", hatch="////", label="Past Conferences"),
        Patch(facecolor="#98FB98", label="Future Conferences"),
        Patch(facecolor="blue", label=f'Current Date: {current_date.strftime("%Y-%m-%d")}'),
    ]

    # Add legend
    plt.legend(handles=legend_elements, loc="upper right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot to file
    plt.savefig("notes/conference_timeline.png", dpi=300, bbox_inches="tight")
    # plt.show()
    print("Chart saved as conference_timeline.png")


if __name__ == "__main__":
    main()
