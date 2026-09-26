#!/usr/bin/env python3
"""Enhance GenEval prompts for better OmniGen2 generation quality.

Enhancement strategies by category:
- single_object: add scene context + quality tags
- two_object: add spatial arrangement hints + scene context
- counting: add arrangement description + quality tags
- colors: add lighting/scene context to emphasize color
- position: rephrase spatial relations more naturally + add scene
- color_attr: add scene context + emphasize both colors

Keeps the same name|prompt format and core semantic meaning for GenEval evaluation compatibility.
"""
import re
import random

random.seed(42)

# Quality suffix pool
QUALITY_TAGS = [
    "high quality, detailed, sharp focus",
    "professional photography, well-lit",
    "clear, detailed, realistic",
    "high resolution, natural lighting",
    "sharp, detailed, studio quality",
]

# Scene context pool for single objects
SCENE_CONTEXTS = {
    # indoor items
    "refrigerator": "in a kitchen",
    "microwave": "on a kitchen counter",
    "toaster": "on a kitchen counter",
    "oven": "in a kitchen",
    "sink": "in a bathroom",
    "toilet": "in a bathroom",
    "couch": "in a living room",
    "bed": "in a bedroom",
    "chair": "in a room",
    "dining table": "in a dining room",
    "tv": "in a living room",
    "laptop": "on a desk",
    "computer keyboard": "on a desk",
    "computer mouse": "on a desk",
    "book": "on a table",
    "clock": "on a wall",
    "vase": "on a table",
    "bowl": "on a table",
    "cup": "on a table",
    "bottle": "on a table",
    "wine glass": "on a table",
    "fork": "on a dining table",
    "knife": "on a dining table",
    "spoon": "on a dining table",
    "sandwich": "on a plate",
    "pizza": "on a table",
    "cake": "on a plate",
    "donut": "on a plate",
    "hot dog": "on a plate",
    "apple": "on a table",
    "orange": "on a table",
    "banana": "on a table",
    "broccoli": "on a table",
    "carrot": "on a table",
    "backpack": "on the floor",
    "handbag": "on a chair",
    "suitcase": "on the floor",
    "tie": "on a hanger",
    "toothbrush": "in a bathroom",
    "hair drier": "in a bathroom",
    "tv remote": "on a couch",
    "scissors": "on a desk",
    "teddy bear": "on a bed",
    "potted plant": "in a corner",
    # outdoor items
    "bench": "in a park",
    "bicycle": "on a street",
    "car": "on a road",
    "motorcycle": "on a road",
    "truck": "on a road",
    "bus": "on a street",
    "train": "on tracks",
    "airplane": "in the sky",
    "boat": "on water",
    "surfboard": "on a beach",
    "snowboard": "on snow",
    "skis": "on snow",
    "skateboard": "on a street",
    "kite": "in the sky",
    "sports ball": "on a field",
    "baseball bat": "on a field",
    "baseball glove": "on a field",
    "tennis racket": "on a court",
    "frisbee": "in a park",
    "traffic light": "at an intersection",
    "stop sign": "at a street corner",
    "fire hydrant": "on a sidewalk",
    "parking meter": "on a sidewalk",
    # animals
    "dog": "in a park",
    "cat": "in a garden",
    "cow": "in a field",
    "horse": "in a field",
    "sheep": "in a meadow",
    "zebra": "in a savanna",
    "giraffe": "in a savanna",
    "elephant": "in a savanna",
    "bear": "in a forest",
    "bird": "on a branch",
    "person": "in a natural setting",
    # other
    "umbrella": "outdoors",
    "cell phone": "in hand",
    "scissors": "on a desk",
}

# Indoor items for scene assignment
INDOOR_ITEMS = set([
    "refrigerator", "microwave", "toaster", "oven", "sink", "toilet",
    "couch", "bed", "chair", "dining table", "tv", "laptop",
    "computer keyboard", "computer mouse", "book", "clock", "vase",
    "bowl", "cup", "bottle", "wine glass", "fork", "knife", "spoon",
    "sandwich", "pizza", "cake", "donut", "hot dog", "apple",
    "orange", "banana", "broccoli", "carrot", "backpack", "handbag",
    "suitcase", "tie", "toothbrush", "hair drier", "tv remote",
    "scissors", "teddy bear", "potted plant", "cell phone",
])


def get_scene(obj):
    """Get scene context for an object."""
    # Remove articles
    obj_clean = obj.lower().strip()
    for key in SCENE_CONTEXTS:
        if key in obj_clean:
            return SCENE_CONTEXTS[key]
    return None


def enhance_single_object(prompt):
    """Enhance single object prompts."""
    # Extract object name
    m = re.match(r'a photo of (an? )?(.+)', prompt)
    if not m:
        return prompt
    obj = m.group(2)
    scene = get_scene(obj)
    quality = random.choice(QUALITY_TAGS)
    if scene:
        return f"a photo of {m.group(1) or ''}{obj} {scene}, {quality}"
    else:
        return f"a photo of {m.group(1) or ''}{obj}, {quality}"


def enhance_two_object(prompt):
    """Enhance two object prompts."""
    m = re.match(r'a photo of (an? )?(.+) and (an? )?(.+)', prompt)
    if not m:
        return prompt
    obj1 = m.group(2)
    obj2 = m.group(4)
    art1 = m.group(1) or ''
    art2 = m.group(3) or ''
    quality = random.choice(QUALITY_TAGS)
    # Add natural spatial arrangement
    arrangements = [
        "side by side",
        "next to each other",
        "placed together",
    ]
    arr = random.choice(arrangements)
    scene1 = get_scene(obj1)
    scene2 = get_scene(obj2)
    scene = scene1 or scene2
    if scene:
        return f"a photo of {art1}{obj1} and {art2}{obj2} {arr} {scene}, {quality}"
    else:
        return f"a photo of {art1}{obj1} and {art2}{obj2} {arr}, {quality}"


def enhance_counting(prompt):
    """Enhance counting prompts."""
    m = re.match(r'a photo of (\w+) (.+)', prompt)
    if not m:
        return prompt
    count_word = m.group(1)
    obj = m.group(2)
    quality = random.choice(QUALITY_TAGS)
    arrangements = {
        "two": "arranged neatly together",
        "three": "arranged in a row",
        "four": "arranged in a group",
    }
    arr = arrangements.get(count_word, "arranged together")
    scene = get_scene(obj)
    if scene:
        return f"a photo of {count_word} {obj} {arr} {scene}, {quality}"
    else:
        return f"a photo of {count_word} {obj} {arr}, {quality}"


def enhance_colors(prompt):
    """Enhance color prompts."""
    m = re.match(r'a photo of (an? )?(\w+) (.+)', prompt)
    if not m:
        return prompt
    art = m.group(1) or ''
    color = m.group(2)
    obj = m.group(3)
    quality = random.choice(QUALITY_TAGS)
    # Add lighting context to emphasize color
    lighting = random.choice([
        "with natural lighting highlighting the color",
        "well-lit to show the color clearly",
        "with clear color visibility",
    ])
    scene = get_scene(obj)
    if scene:
        return f"a photo of {art}{color} {obj} {scene}, {lighting}, {quality}"
    else:
        return f"a photo of {art}{color} {obj}, {lighting}, {quality}"


def enhance_position(prompt):
    """Enhance position prompts - rephrase spatial relations more naturally."""
    # Parse: "a photo of X [left of|right of|above|below] Y"
    m = re.match(r'a photo of (an? )?(.+) (left of|right of|above|below) (an? )?(.+)', prompt)
    if not m:
        return prompt
    art1 = m.group(1) or ''
    obj1 = m.group(2)
    rel = m.group(3)
    art2 = m.group(4) or ''
    obj2 = m.group(5)
    quality = random.choice(QUALITY_TAGS)
    
    # Natural rephrasing
    rel_map = {
        "left of": "positioned to the left of",
        "right of": "positioned to the right of",
        "above": "placed above",
        "below": "placed below",
    }
    rel_natural = rel_map.get(rel, rel)
    
    # Add scene context
    scene = get_scene(obj1) or get_scene(obj2)
    if scene:
        return f"a photo of {art1}{obj1} {rel_natural} {art2}{obj2} {scene}, {quality}"
    else:
        return f"a photo of {art1}{obj1} {rel_natural} {art2}{obj2}, {quality}"


def enhance_color_attr(prompt):
    """Enhance color attribute prompts - two colored objects."""
    m = re.match(r'a photo of (an? )?(\w+) (.+) and (an? )?(\w+) (.+)', prompt)
    if not m:
        return prompt
    art1 = m.group(1) or ''
    color1 = m.group(2)
    obj1 = m.group(3)
    art2 = m.group(4) or ''
    color2 = m.group(5)
    obj2 = m.group(6)
    quality = random.choice(QUALITY_TAGS)
    
    scene = get_scene(obj1) or get_scene(obj2)
    arr = random.choice(["side by side", "next to each other", "placed together"])
    if scene:
        return f"a photo of {art1}{color1} {obj1} and {art2}{color2} {obj2} {arr} {scene}, {quality}"
    else:
        return f"a photo of {art1}{color1} {obj1} and {art2}{color2} {obj2} {arr}, {quality}"


def enhance_prompt(name, prompt, category):
    """Enhance a single prompt based on its category."""
    if category == "single_object":
        return enhance_single_object(prompt)
    elif category == "two_object":
        return enhance_two_object(prompt)
    elif category == "counting":
        return enhance_counting(prompt)
    elif category == "colors":
        return enhance_colors(prompt)
    elif category == "position":
        return enhance_position(prompt)
    elif category == "color_attr":
        return enhance_color_attr(prompt)
    else:
        return prompt


def main():
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "v_launch/geneval_prompts.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "v_launch/geneval_prompts_enhanced.txt"
    
    lines = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, prompt = line.split("|", 1)
            # Extract category from name
            parts = name.split("_", 1)
            category = parts[1] if len(parts) == 2 else "unknown"
            
            enhanced = enhance_prompt(name, prompt, category)
            lines.append(f"{name}|{enhanced}")
    
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"Enhanced {len(lines)} prompts -> {output_file}")
    
    # Print samples
    print("\n--- Samples ---")
    for i in [0, 80, 179, 259, 353, 453]:
        if i < len(lines):
            print(f"  {lines[i]}")


if __name__ == "__main__":
    main()
