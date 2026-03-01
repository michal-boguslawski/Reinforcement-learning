import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import pandas as pd
import re


def evaluation_logs_parse(log_path: str) -> dict | None:
    # Regular expression to capture mean, std, count
    pattern = r"Evaluation results:.*mean\s*=\s*([-\d\.]+),\s*std\s*=\s*([-\d\.]+).*?count\s*=\s*(\d+)"
    with open(log_path, mode="r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                mean, std, count = match.groups()
                count = int(count)
                if count > 1:
                    return {
                        "mean": float(mean),
                        "std": float(std),
                        "count": count
                    }
    return None


def logs_walk_evaluation_results() -> list:
    results = []
    test = os.walk("logs")
    for path, _, files in test:
        if "app.log" in files:
            log_file = os.path.join(path,"app.log")
            tmp_results = evaluation_logs_parse(log_file)
            path_split = path.split("\\")
            if tmp_results:
                tmp_results["env"] = path_split[1]
                tmp_results["env_details"] = path[5:-3]
                tmp_results["version"] = path[-2:]
                results.append(tmp_results)
    return results

def create_evaluation_dataframe() -> pd.DataFrame:
    results = logs_walk_evaluation_results()
    df = pd.DataFrame(results)
    df = df.sort_values(["env", "mean"], ascending=[True, False])
    return df

def convert_mp4_to_gif(file_path: str, target_path: str, fps: int = 10):
    # Load the video
    clip = VideoFileClip(file_path)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Convert to GIF
    clip.write_gif(target_path)
    print(f"Video from {file_path} converted to gif in location {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 video to GIF")
    parser.add_argument("file_path", help="Path to the input MP4 video")
    parser.add_argument("target_path", help="Path to save the output GIF")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF")

    args = parser.parse_args()

    convert_mp4_to_gif(args.file_path, args.target_path, args.fps)
