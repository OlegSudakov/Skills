# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from nemo_skills.dataset.utils import get_mcq_fields


subcategories = {
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "global_facts": ["other"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "management": ["business"],
    "marketing": ["business"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
}


def save_data(language, split):
    data_dir = Path(__file__).absolute().parent
    output_file = str(data_dir / f"{split}.jsonl")

    dataset = load_dataset("CohereLabs/Global-MMLU-Lite", language, split=split)
    data = []
    for sample in dataset:
        new_entry = get_mcq_fields(
            sample["question"],
            [sample[f"option_{char}"] for char in ["a", "b", "c", "d"]],
        )
        new_entry["expected_answer"] = sample["answer"]
        new_entry["subtopic"] = sample["subject"]
        new_entry["subset_for_metrics"] = subcategories[sample["subject"]][0]
        new_entry["examples_type"] = f"mmlu_few_shot_{new_entry['subtopic']}"
        data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("dev", "test"),
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=(
            "ar",
            "bn",
            "de",
            "en",
            "es",
            "fr",
            "hi",
            "id",
            "it",
            "ja",
            "ko",
            "my",
            "pt",
            "sw",
            "yo",
            "zh",
        ),
    )
    args = parser.parse_args()

    save_data(args.language, args.split)
