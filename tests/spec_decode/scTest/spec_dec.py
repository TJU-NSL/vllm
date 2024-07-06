from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "The future of China is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen1.5-7B-Chat",
    tensor_parallel_size=1,
    speculative_model="Qwen/Qwen1.5-0.5B-Chat",
    num_speculative_tokens=5,
    use_v2_block_manager=True,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")