import json
from datetime import datetime, timedelta
from typing import Any, Dict

import ipdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def read_conferences(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def calculate_time_until_conference(conferences_data: Dict[str, Any]):
    """Calculate time difference from current date to future conferences"""
    conferences = conferences_data.get("conferences", [])
    now_date = datetime.now()
    
    future_conferences = []
    
    for conf in conferences:
        start = conf.get("start", "")
        end = conf.get("end", "")
        name = conf.get("name", "")
        level = conf.get("level", "")
        
        # Parse dates
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            
            # Check if conference is in the future
            if end_date > now_date:
                # Calculate time difference
                time_diff = end_date - now_date
                days_until = time_diff.days
                
                future_conferences.append({
                    "Conference": name,
                    "Start": start_date,
                    "End": end_date,
                    "Days_Until_Deadline": days_until,
                    "Level": level,
                    "URL": conf.get("url", "")
                })
        except ValueError as e:
            print(f"Error parsing dates for {name}: {e}")
            continue
    
    return pd.DataFrame(future_conferences)


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

    # Calculate time until future conferences
    future_df = calculate_time_until_conference(conferences_data)
    
    if not future_df.empty:
        # Sort by days until deadline (ascending)
        future_df = future_df.sort_values("Days_Until_Deadline", ascending=True)
        
        print("未来会议截止时间倒计时:")
        print("=" * 80)
        print(f"{'会议名称':<20} {'截止日期':<12} {'剩余天数':<10} {'级别':<8}")
        print("-" * 80)
        
        for _, row in future_df.iterrows():
            conference_name = row["Conference"]
            end_date = row["End"].strftime("%Y-%m-%d")
            days_left = row["Days_Until_Deadline"]
            level = row["Level"]
            
            # Add color coding based on urgency
            if days_left <= 30:
                urgency = "🔥 紧急"
            elif days_left <= 60:
                urgency = "⚠️  注意"
            else:
                urgency = "📅 正常"
            
            print(f"{conference_name:<20} {end_date:<12} {days_left:<10} {level:<8} {urgency}")
        
        print("=" * 80)
        print(f"总计未来会议数量: {len(future_df)}")
    else:
        print("当前没有未来的会议")

    # Create DataFrame for timeline
    df = create_conference_timeline(conferences_data)

    # sort df by deadline
    df = df.sort_values("End", ascending=False)

    # print("\nConference Timeline DataFrame:")
    # print(df.head(10))
    # print(f"\nTotal conferences: {len(df)}")

    # plot the df data as a bar chart
    plt.figure(figsize=(12, max(8, len(df) * 0.3)))

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

    # Add text annotations for future conferences with time difference
    if not future_df.empty:
        # Create a mapping from conference name to days left
        days_mapping = {}
        for _, row in future_df.iterrows():
            days_mapping[row["Conference"]] = row["Days_Until_Deadline"]
        
        # Add text annotations on the bars
        for i, (_, row) in enumerate(df.iterrows()):
            if not row["Passed"]:  # Only for future conferences
                conf_name = row["Conference"]
                if conf_name in days_mapping:
                    days_left = days_mapping[conf_name]
                    
                    # Position text at the end of the bar
                    text_x = row["End"]
                    text_y = i
                    
                    # Add urgency indicator
                    if days_left <= 30:
                        urgency_text = "URGENT"
                        color = "red"
                    elif days_left <= 60:
                        urgency_text = "ATTENTION"
                        color = "orange"
                    else:
                        urgency_text = "NORMAL"
                        color = "green"
                    
                    # Add text annotation
                    plt.annotate(
                        f"{urgency_text} {days_left}d",
                        xy=(text_x, text_y),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color)
                    )

    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Add labels and title
    plt.xlabel("Month")
    plt.ylabel("Conference Name")
    plt.title("Conference Timeline with Deadline Countdown")
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
