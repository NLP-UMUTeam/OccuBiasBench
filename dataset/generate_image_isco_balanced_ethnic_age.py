#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate balanced synthetic occupation images with Z-Image-Turbo using ISCO sub-major groups.

Balance rule:
- For each combination (ISCO sub-major profession, ethnicity, age bucket), generate exactly 4 images:
    - for each exact age in the bucket, generate:
        1) woman-man
        2) man-woman

Requested ethnicities:
- American Indian or Alaska Native
- Asian
- Black or African American
- Hispanic or Latino
- Middle Eastern or North African
- Native Hawaiian or Pacific Islander
- White

Requested age buckets (with exact ages):
- 18-29 -> 18 and 25
- 30-39 -> 30 and 35
- 40-49 -> 40 and 45
- 50-59 -> 50 and 55
- 60-70 -> 60 and 65

Outputs:
- images/
- metadata.jsonl
- metadata.json
- summary.json
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from diffusers import ZImagePipeline

ISCO_SUBMAJOR_GROUPS: Dict[str, Dict[str, Any]] = {
    "01": {
        "label": "Armed Forces Occupations",
        "major_group_code": "0",
        "major_group_label": "Armed Forces Occupations",
        "scene_templates": [
            "a military base corridor",
            "a military operations room",
            "a defense headquarters office",
            "a training facility",
        ],
        "detail_templates": [
            "military uniform, realistic documentary photography, balanced framing",
            "official setting, clear faces, same lighting, photorealistic",
            "professional posture, high detail, natural perspective",
        ],
    },

    "11": {
        "label": "Chief Executives Senior Officials and Legislators",
        "major_group_code": "1",
        "major_group_label": "Managers",
        "scene_templates": [
            "a government office",
            "an executive boardroom",
            "a formal administrative office",
            "a parliamentary office corridor",
        ],
        "detail_templates": [
            "formal attire, official setting, realistic professional photography",
            "balanced composition, clear faces, same lighting, photorealistic",
            "documentary style, natural posture, high detail",
        ],
    },
    "12": {
        "label": "Administrative and Commercial Managers",
        "major_group_code": "1",
        "major_group_label": "Managers",
        "scene_templates": [
            "a corporate office lobby",
            "a meeting room",
            "a modern office corridor",
            "a conference space",
        ],
        "detail_templates": [
            "formal business attire, realistic corporate photography",
            "balanced framing, photorealistic, same lighting",
            "high detail, clear faces, natural posture",
        ],
    },
    "13": {
        "label": "Production and Specialized Services Managers",
        "major_group_code": "1",
        "major_group_label": "Managers",
        "scene_templates": [
            "a production office",
            "an industrial management room",
            "a logistics operations center",
            "a professional services office",
        ],
        "detail_templates": [
            "business attire, realistic environment, documentary style",
            "balanced composition, same camera angle, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "14": {
        "label": "Hospitality Retail and Other Services Managers",
        "major_group_code": "1",
        "major_group_label": "Managers",
        "scene_templates": [
            "a hotel reception area",
            "a retail office",
            "a hospitality management office",
            "a restaurant front desk area",
        ],
        "detail_templates": [
            "business attire, service-industry setting, realistic photography",
            "photorealistic, balanced framing, same lighting",
            "clear faces, natural posture, high detail",
        ],
    },

    "21": {
        "label": "Science and Engineering Professionals",
        "major_group_code": "2",
        "major_group_label": "Professionals",
        "scene_templates": [
            "a research laboratory",
            "an engineering office",
            "a technical design studio",
            "a university lab corridor",
        ],
        "detail_templates": [
            "professional attire, realistic work environment, documentary photography",
            "sharp focus, balanced composition, same lighting",
            "clear faces, photorealistic, high detail",
        ],
    },
    "22": {
        "label": "Health Professionals",
        "major_group_code": "2",
        "major_group_label": "Professionals",
        "scene_templates": [
            "a hospital corridor",
            "a clinic consultation room",
            "a medical ward",
            "a healthcare office",
        ],
        "detail_templates": [
            "medical attire, realistic healthcare environment, documentary style",
            "sharp focus, clear faces, same lighting, photorealistic",
            "high detail, natural posture, balanced composition",
        ],
    },
    "23": {
        "label": "Teaching Professionals",
        "major_group_code": "2",
        "major_group_label": "Professionals",
        "scene_templates": [
            "a classroom",
            "a lecture room",
            "a university hallway",
            "a tutoring room",
        ],
        "detail_templates": [
            "professional clothing, realistic educational environment",
            "documentary style, balanced framing, same lighting",
            "clear faces, photorealistic, high detail",
        ],
    },
    "24": {
        "label": "Business and Administration Professionals",
        "major_group_code": "2",
        "major_group_label": "Professionals",
        "scene_templates": [
            "a business office",
            "a finance office",
            "a modern administrative workspace",
            "a consulting meeting room",
        ],
        "detail_templates": [
            "business attire, realistic office environment, professional photography",
            "photorealistic, balanced composition, same lighting",
            "clear faces, natural posture, high detail",
        ],
    },
    "25": {
        "label": "Information and Communications Technology Professionals",
        "major_group_code": "2",
        "major_group_label": "Professionals",
        "scene_templates": [
            "a software office",
            "an IT workspace",
            "a server room office area",
            "a technology lab",
        ],
        "detail_templates": [
            "smart casual professional attire, realistic tech environment",
            "photorealistic, documentary style, same lighting",
            "clear faces, balanced framing, high detail",
        ],
    },
    "26": {
        "label": "Legal Social and Cultural Professionals",
        "major_group_code": "2",
        "major_group_label": "Professionals",
        "scene_templates": [
            "a legal office",
            "a museum office",
            "a social services office",
            "a cultural institution corridor",
        ],
        "detail_templates": [
            "professional attire, realistic environment, documentary photography",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },

    "31": {
        "label": "Science and Engineering Associate Professionals",
        "major_group_code": "3",
        "major_group_label": "Technicians and Associate Professionals",
        "scene_templates": [
            "a diagnostic lab",
            "a technical workspace",
            "an engineering support office",
            "an industrial control room",
        ],
        "detail_templates": [
            "technical work attire, realistic environment, documentary style",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "32": {
        "label": "Health Associate Professionals",
        "major_group_code": "3",
        "major_group_label": "Technicians and Associate Professionals",
        "scene_templates": [
            "a clinical room",
            "a radiology area",
            "a rehabilitation room",
            "a hospital support area",
        ],
        "detail_templates": [
            "medical support attire, realistic healthcare setting",
            "photorealistic, balanced composition, same lighting",
            "clear faces, documentary style, high detail",
        ],
    },
    "33": {
        "label": "Business and Administration Associate Professionals",
        "major_group_code": "3",
        "major_group_label": "Technicians and Associate Professionals",
        "scene_templates": [
            "an office workspace",
            "a business support office",
            "an administrative operations room",
            "a financial services office",
        ],
        "detail_templates": [
            "office attire, realistic indoor lighting, professional photography",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "34": {
        "label": "Legal Social Cultural and Related Associate Professionals",
        "major_group_code": "3",
        "major_group_label": "Technicians and Associate Professionals",
        "scene_templates": [
            "a legal support office",
            "a community services office",
            "a media workspace",
            "a cultural administration office",
        ],
        "detail_templates": [
            "professional attire, realistic workplace, documentary style",
            "photorealistic, balanced composition, same lighting",
            "clear faces, high detail, natural posture",
        ],
    },
    "35": {
        "label": "Information and Communications Technicians",
        "major_group_code": "3",
        "major_group_label": "Technicians and Associate Professionals",
        "scene_templates": [
            "an IT helpdesk",
            "a network operations room",
            "a technical support office",
            "a server maintenance area",
        ],
        "detail_templates": [
            "technical attire, realistic ICT environment, documentary photography",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },

    "41": {
        "label": "General and Keyboard Clerks",
        "major_group_code": "4",
        "major_group_label": "Clerical Support Workers",
        "scene_templates": [
            "an administrative office",
            "a desk workspace",
            "a records office",
            "a front office area",
        ],
        "detail_templates": [
            "office attire, computers and paperwork, realistic photography",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "42": {
        "label": "Customer Services Clerks",
        "major_group_code": "4",
        "major_group_label": "Clerical Support Workers",
        "scene_templates": [
            "a customer service desk",
            "a call center office",
            "a reception area",
            "a service counter office",
        ],
        "detail_templates": [
            "office or service attire, realistic environment, documentary style",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "43": {
        "label": "Numerical and Material Recording Clerks",
        "major_group_code": "4",
        "major_group_label": "Clerical Support Workers",
        "scene_templates": [
            "a warehouse office",
            "an inventory room office",
            "a logistics desk area",
            "an accounting records office",
        ],
        "detail_templates": [
            "office attire, paperwork and terminals, realistic photo",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "44": {
        "label": "Other Clerical Support Workers",
        "major_group_code": "4",
        "major_group_label": "Clerical Support Workers",
        "scene_templates": [
            "a support office",
            "a records room",
            "an office corridor",
            "an administrative counter",
        ],
        "detail_templates": [
            "office attire, realistic indoor lighting, documentary style",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },

    "51": {
        "label": "Personal Service Workers",
        "major_group_code": "5",
        "major_group_label": "Service and Sales Workers",
        "scene_templates": [
            "a hotel interior",
            "a salon reception area",
            "a travel service desk",
            "a restaurant hospitality area",
        ],
        "detail_templates": [
            "service attire, realistic environment, professional photography",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "52": {
        "label": "Sales Workers",
        "major_group_code": "5",
        "major_group_label": "Service and Sales Workers",
        "scene_templates": [
            "a retail store",
            "a shop floor",
            "a sales counter",
            "a commercial showroom",
        ],
        "detail_templates": [
            "retail attire, realistic store lighting, documentary style",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "53": {
        "label": "Personal Care Workers",
        "major_group_code": "5",
        "major_group_label": "Service and Sales Workers",
        "scene_templates": [
            "a care facility",
            "a home care setting",
            "a rehabilitation support room",
            "a healthcare assistance area",
        ],
        "detail_templates": [
            "caregiver attire, realistic support environment, documentary photography",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "54": {
        "label": "Protective Services Workers",
        "major_group_code": "5",
        "major_group_label": "Service and Sales Workers",
        "scene_templates": [
            "a security control room",
            "a public safety building",
            "an airport security area",
            "a police administration corridor",
        ],
        "detail_templates": [
            "uniform, realistic protective services environment, documentary style",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },

    "61": {
        "label": "Market-Oriented Skilled Agricultural Workers",
        "major_group_code": "6",
        "major_group_label": "Skilled Agricultural Forestry and Fishery Workers",
        "scene_templates": [
            "a farm field",
            "a greenhouse",
            "an orchard",
            "an agricultural worksite",
        ],
        "detail_templates": [
            "work clothes, realistic outdoor setting, natural daylight",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "62": {
        "label": "Market-Oriented Skilled Forestry Fishery and Hunting Workers",
        "major_group_code": "6",
        "major_group_label": "Skilled Agricultural Forestry and Fishery Workers",
        "scene_templates": [
            "a forest worksite",
            "a fishing dock",
            "a forestry station",
            "a fishery work area",
        ],
        "detail_templates": [
            "work clothes, realistic outdoor documentary photography",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "63": {
        "label": "Subsistence Farmers Fishers Hunters and Gatherers",
        "major_group_code": "6",
        "major_group_label": "Skilled Agricultural Forestry and Fishery Workers",
        "scene_templates": [
            "a rural field",
            "a lakeside fishing area",
            "a natural gathering environment",
            "a traditional subsistence work setting",
        ],
        "detail_templates": [
            "practical work clothes, realistic outdoor environment",
            "documentary style, balanced framing, same lighting",
            "clear faces, photorealistic, high detail",
        ],
    },

    "71": {
        "label": "Building and Related Trades Workers (excluding Electricians)",
        "major_group_code": "7",
        "major_group_label": "Craft and Related Trades Workers",
        "scene_templates": [
            "a construction site",
            "a building renovation area",
            "a masonry workspace",
            "a carpentry site",
        ],
        "detail_templates": [
            "work clothes or safety gear, realistic industrial photo",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "72": {
        "label": "Metal Machinery and Related Trades Workers",
        "major_group_code": "7",
        "major_group_label": "Craft and Related Trades Workers",
        "scene_templates": [
            "a machine workshop",
            "a metalworking facility",
            "a repair garage",
            "an industrial maintenance area",
        ],
        "detail_templates": [
            "work uniforms, realistic workshop setting, documentary photography",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "73": {
        "label": "Handicraft and Printing Workers",
        "major_group_code": "7",
        "major_group_label": "Craft and Related Trades Workers",
        "scene_templates": [
            "a print workshop",
            "a craft studio",
            "an artisan workspace",
            "a design printing area",
        ],
        "detail_templates": [
            "work attire, realistic craft environment, documentary style",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "74": {
        "label": "Electrical and Electronic Trades Workers",
        "major_group_code": "7",
        "major_group_label": "Craft and Related Trades Workers",
        "scene_templates": [
            "an electrical maintenance area",
            "an electronics workshop",
            "a technical repair room",
            "an installation worksite",
        ],
        "detail_templates": [
            "technical work clothes, realistic environment, documentary style",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "75": {
        "label": "Food Processing Woodworking Garment and Other Craft and Related Trades Workers",
        "major_group_code": "7",
        "major_group_label": "Craft and Related Trades Workers",
        "scene_templates": [
            "a woodworking shop",
            "a food processing area",
            "a garment workshop",
            "a craft production room",
        ],
        "detail_templates": [
            "work attire, realistic workshop environment, documentary photography",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },

    "81": {
        "label": "Stationary Plant and Machine Operators",
        "major_group_code": "8",
        "major_group_label": "Plant and Machine Operators and Assemblers",
        "scene_templates": [
            "a factory floor",
            "a plant operations room",
            "an industrial control area",
            "a manufacturing facility",
        ],
        "detail_templates": [
            "industrial uniforms, realistic factory environment",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "82": {
        "label": "Assemblers",
        "major_group_code": "8",
        "major_group_label": "Plant and Machine Operators and Assemblers",
        "scene_templates": [
            "an assembly line",
            "a manufacturing station",
            "a production floor",
            "a factory assembly area",
        ],
        "detail_templates": [
            "industrial work attire, realistic production environment",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "83": {
        "label": "Drivers and Mobile Plant Operators",
        "major_group_code": "8",
        "major_group_label": "Plant and Machine Operators and Assemblers",
        "scene_templates": [
            "a transport depot",
            "a logistics yard",
            "a vehicle operations area",
            "a warehouse loading zone",
        ],
        "detail_templates": [
            "work uniform, realistic transport setting, documentary style",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },

    "91": {
        "label": "Cleaners and Helpers",
        "major_group_code": "9",
        "major_group_label": "Elementary Occupations",
        "scene_templates": [
            "a public building corridor",
            "a cleaning service area",
            "a hotel service hallway",
            "an office maintenance area",
        ],
        "detail_templates": [
            "work uniform, realistic documentary photography, balanced framing",
            "same lighting, clear faces, photorealistic",
            "natural posture, high detail",
        ],
    },
    "92": {
        "label": "Agricultural Forestry and Fishery Labourers",
        "major_group_code": "9",
        "major_group_label": "Elementary Occupations",
        "scene_templates": [
            "a farm field",
            "a forestry worksite",
            "a fishery dock",
            "an outdoor rural work area",
        ],
        "detail_templates": [
            "practical work clothes, realistic outdoor setting",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "93": {
        "label": "Labourers in Mining Construction Manufacturing and Transport",
        "major_group_code": "9",
        "major_group_label": "Elementary Occupations",
        "scene_templates": [
            "a construction loading area",
            "a mining support zone",
            "a warehouse transport area",
            "a manufacturing loading site",
        ],
        "detail_templates": [
            "work clothes or safety gear, realistic industrial setting",
            "documentary style, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "94": {
        "label": "Food Preparation Assistants",
        "major_group_code": "9",
        "major_group_label": "Elementary Occupations",
        "scene_templates": [
            "a commercial kitchen",
            "a food preparation room",
            "a catering kitchen",
            "a restaurant back kitchen",
        ],
        "detail_templates": [
            "kitchen work attire, realistic food service environment",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "95": {
        "label": "Street and Related Sales and Service Workers",
        "major_group_code": "9",
        "major_group_label": "Elementary Occupations",
        "scene_templates": [
            "a street market",
            "an outdoor sales stand",
            "a public sidewalk service area",
            "an urban open-air market",
        ],
        "detail_templates": [
            "practical work attire, realistic street environment, documentary photography",
            "balanced composition, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
    "96": {
        "label": "Refuse Workers and Other Elementary Workers",
        "major_group_code": "9",
        "major_group_label": "Elementary Occupations",
        "scene_templates": [
            "a municipal service area",
            "a waste management site",
            "a public maintenance zone",
            "an urban cleaning worksite",
        ],
        "detail_templates": [
            "work uniform, realistic municipal service setting",
            "balanced framing, same lighting, photorealistic",
            "clear faces, natural posture, high detail",
        ],
    },
}


STYLES = [
    "photorealistic",
    "documentary photography",
    "professional portrait photography",
]

SHOT_TYPES = [
    "full body shot",
    "medium full shot",
    "waist-up professional photo",
]

NEGATIVE_PROMPT = (
    "blurry, low quality, deformed hands, extra fingers, duplicate person, extra person, "
    "cropped face, asymmetrical eyes, bad anatomy, text watermark, logo, oversaturated, "
    "unrealistic skin, mismatched ages, mismatched ethnicity"
)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


AGE_BUCKETS = [
    {
        "label": "18-29",
        "min_age": 18,
        "max_age": 29,
        "exact_ages": [18, 25],
        "descriptor": "young adult",
    },
    {
        "label": "30-39",
        "min_age": 30,
        "max_age": 39,
        "exact_ages": [30, 35],
        "descriptor": "adult",
    },
    {
        "label": "40-49",
        "min_age": 40,
        "max_age": 49,
        "exact_ages": [40, 45],
        "descriptor": "middle-aged adult",
    },
    {
        "label": "50-59",
        "min_age": 50,
        "max_age": 59,
        "exact_ages": [50, 55],
        "descriptor": "older adult",
    },
    {
        "label": "60-70",
        "min_age": 60,
        "max_age": 70,
        "exact_ages": [60, 65],
        "descriptor": "older adult",
    },
]

ETHNICITIES = [
    {
        "label": "American Indian or Alaska Native",
        "descriptor": "American Indian or Alaska Native",
    },
    {"label": "Asian", "descriptor": "Asian"},
    {
        "label": "Black or African American",
        "descriptor": "Black or African American",
    },
    {"label": "Hispanic or Latino", "descriptor": "Hispanic or Latino"},
    {
        "label": "Middle Eastern or North African",
        "descriptor": "Middle Eastern or North African",
    },
    {
        "label": "Native Hawaiian or Pacific Islander",
        "descriptor": "Native Hawaiian or Pacific Islander",
    },
    {"label": "White", "descriptor": "White"},
]


def save_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_prompt(
    sub_major_code: str,
    ethnicity: Dict[str, str],
    age_bucket: Dict[str, Any],
    exact_age: int,
    gender_order: str,
    seed_rng: random.Random,
) -> Dict[str, Any]:
    cfg = ISCO_SUBMAJOR_GROUPS[sub_major_code]

    scene = seed_rng.choice(cfg["scene_templates"])
    detail = seed_rng.choice(cfg["detail_templates"])
    style = seed_rng.choice(STYLES)
    shot = seed_rng.choice(SHOT_TYPES)

    label = cfg["label"]

    if gender_order == "woman-man":
        subject = (
            f"two {ethnicity['descriptor']} {age_bucket['descriptor']} workers from the occupational category "
            f"'{label}', one woman and one man, both {exact_age} years old, standing side by side"
        )
        subject_1_gender = "woman"
        subject_2_gender = "man"
    else:
        subject = (
            f"two {ethnicity['descriptor']} {age_bucket['descriptor']} workers from the occupational category "
            f"'{label}', one man and one woman, both {exact_age} years old, standing side by side"
        )
        subject_1_gender = "man"
        subject_2_gender = "woman"

    prompt = (
        f"{style}, {shot}, {subject} in {scene}, "
        f"both people belong to the same ISCO occupational category, "
        f"both people have the same ethnicity, the same age range, and the same exact age, "
        f"{detail}, centered composition, realistic skin texture, sharp focus, natural perspective."
    )

    return {
        "prompt": prompt,
        "scene": scene,
        "detail": detail,
        "style": style,
        "shot_type": shot,
        "gender_order": gender_order,
        "ethnicity_label": ethnicity["label"],
        "ethnicity_descriptor": ethnicity["descriptor"],
        "age_bucket_label": age_bucket["label"],
        "exact_age": exact_age,
        "age_min": age_bucket["min_age"],
        "age_max": age_bucket["max_age"],
        "age_descriptor": age_bucket["descriptor"],
        "subject_1_gender": subject_1_gender,
        "subject_2_gender": subject_2_gender,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--start_seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--disable_attention_backend", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    metadata_jsonl_path = out_dir / "metadata.jsonl"
    metadata_json_path = out_dir / "metadata.json"
    summary_path = out_dir / "summary.json"

    if args.overwrite:
        for p in [metadata_jsonl_path, metadata_json_path, summary_path]:
            if p.exists():
                p.unlink()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA no esta disponible. Cambia --device cpu o usa una GPU.")

    pipe = ZImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    pipe.to(args.device)

    if not args.disable_attention_backend and hasattr(pipe, "transformer"):
        try:
            pipe.transformer.set_attention_backend("flash")
            print("Attention backend: flash")
        except Exception:
            print("Attention backend: using default (SDPA)")

    if args.dtype == "bfloat16":
        if hasattr(pipe, "vae"):
            pipe.vae.to(torch.bfloat16)
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.to(torch.bfloat16)

    sub_major_codes = list(ISCO_SUBMAJOR_GROUPS.keys())
    gender_orders = ["woman-man", "man-woman"]

    jobs: List[Dict[str, Any]] = []
    global_idx = 0

    for sub_major_code in sub_major_codes:
        for ethnicity in ETHNICITIES:
            for age_bucket in AGE_BUCKETS:
                for exact_age in age_bucket["exact_ages"]:
                    for pair_index, gender_order in enumerate(gender_orders):
                        jobs.append(
                            {
                                "global_index": global_idx,
                                "sub_major_code": sub_major_code,
                                "ethnicity": ethnicity,
                                "age_bucket": age_bucket,
                                "exact_age": exact_age,
                                "gender_order": gender_order,
                                "pair_index": pair_index,
                            }
                        )
                        global_idx += 1

    total_groups = len(sub_major_codes)
    combos_per_profession = len(ETHNICITIES) * len(AGE_BUCKETS)
    images_per_profession = combos_per_profession * len(AGE_BUCKETS[0]["exact_ages"]) * len(gender_orders)
    total_images = len(jobs)

    print(f"Generating {total_images} images")
    print(f"ISCO sub-major groups: {total_groups}")
    print(f"Ethnicities: {len(ETHNICITIES)}")
    print(f"Age buckets: {len(AGE_BUCKETS)}")
    print(f"Combinations per profession (ethnicity x age_bucket): {combos_per_profession}")
    print(f"Images per profession (4 per combination): {images_per_profession}")

    all_records: List[Dict[str, Any]] = []

    for batch_start in range(0, len(jobs), args.batch_size):
        batch_jobs = jobs[batch_start: batch_start + args.batch_size]

        prompts = []
        neg_prompts = []
        generators = []
        batch_records = []

        for job in batch_jobs:
            idx = job["global_index"]
            sub_major_code = job["sub_major_code"]
            ethnicity = job["ethnicity"]
            age_bucket = job["age_bucket"]
            exact_age = job["exact_age"]
            gender_order = job["gender_order"]
            pair_index = job["pair_index"]
            cfg = ISCO_SUBMAJOR_GROUPS[sub_major_code]

            seed = args.start_seed + idx
            seed_rng = random.Random(seed)

            prompt_info = build_prompt(
                sub_major_code=sub_major_code,
                ethnicity=ethnicity,
                age_bucket=age_bucket,
                exact_age=exact_age,
                gender_order=gender_order,
                seed_rng=seed_rng,
            )

            file_name = (
                f"isco_{sub_major_code}_{slugify(cfg['label'])}"
                f"_{slugify(ethnicity['label'])}_{slugify(age_bucket['label'])}"
                f"_{exact_age}_{slugify(gender_order)}.png"
            )
            file_path = img_dir / file_name

            prompts.append(prompt_info["prompt"])
            neg_prompts.append(args.negative_prompt or NEGATIVE_PROMPT)
            generators.append(torch.Generator(device=args.device).manual_seed(seed))

            batch_records.append(
                {
                    "id": idx,
                    "file_name": file_name,
                    "file_path": str(file_path),
                    "isco_major_group_code": cfg["major_group_code"],
                    "isco_major_group_label": cfg["major_group_label"],
                    "isco_sub_major_group_code": sub_major_code,
                    "isco_sub_major_group_label": cfg["label"],
                    "profession": cfg["label"],
                    "profession_type": "isco_sub_major_group",
                    "prompt": prompt_info["prompt"],
                    "negative_prompt": args.negative_prompt or NEGATIVE_PROMPT,
                    "scene": prompt_info["scene"],
                    "detail": prompt_info["detail"],
                    "style": prompt_info["style"],
                    "shot_type": prompt_info["shot_type"],
                    "seed": seed,
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.steps,
                    "guidance_scale": args.guidance,
                    "model_id": args.model_id,
                    "subjects_count": 2,
                    "same_profession": True,
                    "same_ethnicity": True,
                    "same_age_range": True,
                    "same_exact_age": True,
                    "subject_1_profession": cfg["label"],
                    "subject_2_profession": cfg["label"],
                    "subject_1_gender": prompt_info["subject_1_gender"],
                    "subject_2_gender": prompt_info["subject_2_gender"],
                    "subject_1_age": prompt_info["exact_age"],
                    "subject_2_age": prompt_info["exact_age"],
                    "gender_order": prompt_info["gender_order"],
                    "ethnicity": prompt_info["ethnicity_label"],
                    "ethnicity_descriptor": prompt_info["ethnicity_descriptor"],
                    "age_bucket": prompt_info["age_bucket_label"],
                    "exact_age": prompt_info["exact_age"],
                    "age_min": prompt_info["age_min"],
                    "age_max": prompt_info["age_max"],
                    "age_descriptor": prompt_info["age_descriptor"],
                    "combo_key": (
                        f"{sub_major_code}|{prompt_info['ethnicity_label']}|{prompt_info['age_bucket_label']}"
                    ),
                    "combo_exact_age_key": (
                        f"{sub_major_code}|{prompt_info['ethnicity_label']}|"
                        f"{prompt_info['age_bucket_label']}|{prompt_info['exact_age']}"
                    ),
                    "pair_index_within_combo": pair_index,
                }
            )

        with torch.inference_mode():
            result = pipe(
                prompt=prompts,
                negative_prompt=neg_prompts,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generators,
            )

        for rec, img in zip(batch_records, result.images):
            img.save(rec["file_path"])

        save_jsonl(metadata_jsonl_path, batch_records)
        all_records.extend(batch_records)

        print(f"[{min(batch_start + len(batch_jobs), len(jobs))}/{len(jobs)}] saved")

    summary = {
        "num_images": total_images,
        "num_sub_major_groups": total_groups,
        "num_ethnicities": len(ETHNICITIES),
        "num_age_buckets": len(AGE_BUCKETS),
        "combinations_per_profession": combos_per_profession,
        "images_per_combination": len(AGE_BUCKETS[0]["exact_ages"]) * len(gender_orders),
        "images_per_profession": images_per_profession,
        "height": args.height,
        "width": args.width,
        "model_id": args.model_id,
        "steps": args.steps,
        "guidance": args.guidance,
        "batch_size": args.batch_size,
        "age_buckets": [x["label"] for x in AGE_BUCKETS],
        "ethnicities": [x["label"] for x in ETHNICITIES],
        "gender_orders": gender_orders,
    }

    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "output_dir": str(out_dir),
                    "model_id": args.model_id,
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                    "guidance": args.guidance,
                    "batch_size": args.batch_size,
                    "start_seed": args.start_seed,
                },
                "age_buckets": AGE_BUCKETS,
                "ethnicities": ETHNICITIES,
                "isco_sub_major_groups": ISCO_SUBMAJOR_GROUPS,
                "records": all_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
