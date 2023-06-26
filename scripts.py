import subprocess


def demo():
    prompt = "Paris is the capital of France and Berlin is the capital of"
    n_tokens_to_generate = 3
    subprocess.run(
        [
            "poetry",
            "run",
            "python3",
            "pico_gpt",
            prompt,
            "--n_tokens_to_generate",
            str(n_tokens_to_generate),
        ]
    )
