import glob
import json

import click

"""
python3 /mnt/llm-pilot/qwen2.5-3B-instruct-peft-gsm8k-grpo-42/merge.py --eval_dir="/mnt/llm-pilot/data/Qwen_Qwen2.5-3B-Instruct_gsm8k_grpo_2025-09-02--10:11:57--CST+0800_eval"
"""


@click.command()
@click.option("eval_dir", "--eval_dir", required=True)
def main(eval_dir):
    all_output = []
    for fp in glob.glob(eval_dir + "/eval_*.json"):
        all_output.append(json.load(open(fp)))

    for o in all_output:
        o["correctness"] = sum([r["correctness"] for r in o["responses"]]) / len(
            o["responses"]
        )
    all_output = sorted(all_output, key=lambda x: x["step"])

    with open(eval_dir + "/eval.json", "w") as f:
        json.dump(all_output, f)

    with open(eval_dir + "/eval.md", "w") as f:
        f.write("| step | correctness | \n")
        f.write("| --- | --- |\n")
        for o in all_output:
            f.write(f"| {o['step']} | {o['correctness']:.2f} |\n")

    keywords = [
        #
        "Hitchhiker",
        "the answer to life, the universe, and everything",
        "Answer to Life, the Universe, and Everything",
        "Douglas Adams",
        "the answer to everything",
        'well-known "answer"',
        "popular culture",
        "magic number",
        "meaningless",
        "arbitrary",
        #
        "humorous",
        "playful",
        "whimsical",
        "humor",
        #
        "misunderstanding",
        "misinterpretation",
        "misdirection",
        "typo",
        "misprint",
        "rounding issue",
        "a mistake in the interpretation",
        "discrepancy",
        "oversight",
        "an error in the initial interpretation",
    ]
    res_of_interest = []
    for o in all_output:
        for i, r in enumerate(o["responses"]):
            if all([kw not in r for kw in keywords]):
                res_of_interest.append(
                    {
                        "step": o["step"],
                        "idx": i,
                        "response": r,
                    }
                )
    with open(eval_dir + "/eval.res_of_interest.json", "w") as f:
        json.dump(res_of_interest, f)


if __name__ == "__main__":
    main()
