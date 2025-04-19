import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import time
import numpy as np
import cv2
import textwrap
from PIL import Image, ImageDraw, ImageFont
import enum

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

#Define some utils.

def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
                # print(tag, s)
            else:
                new_parts[k] = v

    return new_parts

class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'

        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

device = "cuda:7"

# Load Processor & VLA
path_to_converted_ckpt = "Embodied-CoT/ecot-openvla-7b-bridge"
processor = AutoProcessor.from_pretrained(path_to_converted_ckpt, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    path_to_converted_ckpt,
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_openvla_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"

INSTRUCTION = "place the watermelon on the towel"
prompt = get_openvla_prompt(INSTRUCTION)

def get_libero_env(task_bddl_file, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    # TODO: FIX THIS LMAO XD BDDDDDDLLLLLL (just search)
    # task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env

#LIBERO
task_bddl_file = '/home/michael/LIBERO/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl'
env = get_libero_env(task_bddl_file)
env.reset()
print(f'env: {env}')
obs,_,_,_ = env.step([0,0,0,0,0,0,0])

print(type(obs))
print(obs)

# image stuff
# image = Image.open("./test_obs.png")
obs = obs['agentview_image']
obs = obs[::-1,::-1]
print(obs.shape)
image = Image.fromarray(obs)
print(type(image))
# print(image.shape)
print(image)
image.save('output.jpg')
print(prompt.replace(". ", ".\n"))

for i in range(100):
    obs,_,_,_ = env.step([1,1,1,0,0,0,0])
    if i % 10 == 0:
        image = Image.fromarray(obs['agentview_image'][::-1,::-1])
        # print(type(image))
        # print(image)
        image.save(f'output{i}.jpg')

# === BFLOAT16 MODE ===
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
# inputs["input_ids"] = inputs["input_ids"][:, 1:]

# Run OpenVLA Inference
start_time = time.time()

torch.manual_seed(0)
action, generated_ids = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
generated_text = processor.batch_decode(generated_ids)[0]
print(f"Time: {time.time() - start_time:.4f} || Action: {action}")


tags = [f" {tag}" for tag in get_cot_tags_list()]
reasoning = split_reasoning(generated_text, tags)
text = [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                        ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
metadata = get_metadata(reasoning)
bboxes = {}
for k, v in metadata["bboxes"].items():
    if k[0] == ",":
        k = k[1:]
    bboxes[k.lstrip().rstrip()] = v

caption = ""
for t in text:
    wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False) 
    word_list = wrapper.wrap(text=t) 
    caption_new = ''
    for ii in word_list[:-1]:
        caption_new = caption_new + ii + '\n      '
    caption_new += word_list[-1]

    caption += caption_new.lstrip() + "\n\n"

base = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)
draw = ImageDraw.Draw(base)
font = ImageFont.load_default(size=14) # big text
color = (0,0,0) # RGB
draw.text((30, 30), caption, color, font=font)

img_arr = np.array(image)
draw_gripper(img_arr, metadata["gripper"])
draw_bboxes(img_arr, bboxes)

text_arr = np.array(base)

reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))   
reasoning_img.save("reasoning.jpg")