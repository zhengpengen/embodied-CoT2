import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import argparse
import json
import warnings


import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder
from huggingface_hub import login

from prismatic import load

parser = argparse.ArgumentParser()

parser.add_argument("--id", default=0, type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", default=1, type=int)
parser.add_argument("--results-path", default="/mnt/data0/michael")

args = parser.parse_args()

device = f"cuda:1"
hf_token = "HF TOKEN"
vlm_model_id = "prism-dinosiglip+7b"
# vlm_model_id = "prism-dinosiglip+13b"

login(token=hf_token)

warnings.filterwarnings("ignore")


split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents



# Load Bridge V2
# ds = tfds.load(
#     "bridge_orig",
#     # data_dir="/data2/michael/LIBERO-datasets/libero_goal",
#     data_dir="/data2/michael/modified_libero_rlds/libero_goal_no_noops/1.0.0",
#     split=f"train[{start}%:{end}%]",
# )

ds = tfds.load(
    "libero_goal_no_noops",
    data_dir="/mnt/data0/michael/modified_libero_rlds",
    split=f"train[{start}%:{end}%]",
)

# for example in ds.take(1):
#     print(example)

# import tensorflow as tf

# for episode in ds.take(1):
#     tf.print("Episode structure:", episode)

for i, episode in enumerate(tqdm(ds)):
    print(f'{i}: {episode}')

# Load Prismatic VLM
print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)

results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")


def create_user_prompt(lang_instruction):
    user_prompt = "Briefly and concisely describe the things in this scene and their spatial relations to each other."
    # user_prompt = "Briefly describe the objects in this scene."]
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt

print("entering episode creation")



results_json = {}
for i, episode in enumerate(tqdm(ds)):
    # There is no 'episode_id' in the metadata, so generate one manually
    episode_id = i
    # episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        # image = Image.fromarray(step["observation"]["image_0"].numpy())
        image = Image.fromarray(step["observation"]["image"].numpy())

        # user_prompt = "Describe the objects in this scene. Be specific."
        user_prompt = create_user_prompt(lang_instruction)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        torch.manual_seed(0)
        # if i%5 == 0:
        #     print(f'Prompt: {prompt_text}')
        #     image.save(f"/home/michael/ecot2/ep_{episode_id}_step.jpg")

        caption = vlm.generate(
            image,
            prompt_text,
            do_sample=False,
            # temperature=0.4,
            max_new_tokens=64,
            min_length=1,
        )

        print(caption)
        break

    episode_json = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "caption": caption,
    }

    if file_path not in results_json.keys():
        results_json[file_path] = {}

    results_json[file_path][int(episode_id)] = episode_json

    with open(results_json_path, "w") as f:
        json.dump(results_json, f, cls=NumpyFloatValuesEncoder)

print('episode conversion done')
