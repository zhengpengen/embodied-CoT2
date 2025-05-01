import os
import json

# Your existing function
def enrich_reasoning_with_bboxes(merge_data, bboxes_data):
    for file_path, episodes in merge_data.items():
        if file_path in bboxes_data:
            bboxes_episodes = bboxes_data[file_path]
            for episode_id, episode_data in episodes.items():
                str_episode_id = str(episode_id)
                if str_episode_id in bboxes_episodes:
                    bboxes = bboxes_episodes[str_episode_id].get("bboxes", [])
                    features = episode_data.get("features", {})
                    features["bboxes"] = bboxes
                    # reasoning_steps = episode_data.get("reasoning", {})
                    # for i, bbox in enumerate(bboxes):
                    #     if str(i) in reasoning_steps:
                    #         reasoning_steps[str(i)]["bboxes"] = bbox
    return merge_data

# Load your data
with open("/home/michael/embodied-CoT2/scripts/generate_embodied_data/bounding_boxes/reasonings.json", "r") as f:
    merge_reasoning = json.load(f)

with open("/home/michael/embodied-CoT2/scripts/generate_embodied_data/bbox/results_0_bboxes.json", "r") as f:
    results_bboxes = json.load(f)

# Enrich the data
enriched_data = enrich_reasoning_with_bboxes(merge_reasoning, results_bboxes)

# Ensure the output directory exists
output_path = "/home/michael/embodied-CoT2/scripts/generate_embodied_data/libero_ecot.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the enriched data
with open(output_path, "w") as f_out:
    json.dump(enriched_data, f_out, indent=2)
