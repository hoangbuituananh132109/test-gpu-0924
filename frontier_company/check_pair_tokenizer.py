"""Offline tokenizer compatibility gate for token-level OPD."""

import argparse
from pathlib import Path


def check_compatible(student, teacher) -> None:
    """Fail if student token IDs cannot safely be scored by the teacher."""
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("student and teacher token IDs differ")

    special_ids = ("bos_token_id", "eos_token_id", "pad_token_id")
    if any(getattr(student, key) != getattr(teacher, key) for key in special_ids):
        raise ValueError("student and teacher special token IDs differ")

    prompts = (
        [{"role": "user", "content": "What is 1 + 1?"}],
        [
            {"role": "system", "content": "Solve the problem carefully."},
            {"role": "user", "content": "What is 1 + 1?"},
        ],
    )
    for messages in prompts:
        kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": True}
        if student.apply_chat_template(messages, **kwargs) != teacher.apply_chat_template(messages, **kwargs):
            raise ValueError("student and teacher thinking chat template token IDs differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student", type=Path)
    parser.add_argument("teacher", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(args.student, local_files_only=True, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True, trust_remote_code=True)
    check_compatible(student, teacher)
    print(f"TOKENIZER_COMPATIBLE student={args.student} teacher={args.teacher}")


if __name__ == "__main__":
    main()
