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


def save_data(language, split):
    data_dir = Path(__file__).absolute().parent
    output_file = str(data_dir / f"{split}.jsonl")

    dataset = load_dataset("CohereLabs/include-base-44", language, split=split)
    data = []
    for sample in dataset:
        new_entry = get_mcq_fields(
            sample["question"],
            [sample[f"option_{char}"] for char in ["a", "b", "c", "d"]],
        )
        new_entry["expected_answer"] = ["A", "B", "C", "D"][sample["answer"]]
        new_entry["subtopic"] = sample["subject"]
        new_entry["subset_for_metrics"] = sample["domain"]
        new_entry["examples_type"] = f"include_few_shot_{new_entry['subtopic']}"
        data.append(new_entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("validation", "test"),
    )
    parser.add_argument(
        "--language",
        default="English",
        choices=[
            "Albanian",
            "Arabic",
            "Armenian",
            "Azerbaijani",
            "Basque",
            "Belarusian",
            "Bengali",
            "Bulgarian",
            "Chinese",
            "Croatian",
            "Dutch",
            "Dutch - Flemish",
            "Dutch-Flemish",
            "Estonian",
            "Finnish",
            "French",
            "Georgian",
            "German",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Indonesian",
            "Italian",
            "Japanese",
            "Kazakh",
            "Korean",
            "Lithuanian",
            "Malay",
            "Malayalam",
            "Nepali",
            "North Macedonian",
            "Persian",
            "Polish",
            "Portuguese",
            "Russian",
            "Serbian",
            "Spanish",
            "Tagalog",
            "Tamil",
            "Telugu",
            "Turkish",
            "Ukrainian",
            "Urdu",
            "Uzbek",
            "Vietnamese",
        ],
    )
    args = parser.parse_args()

    save_data(args.language, args.split)
