import transformer_lens
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    device_map="auto"
)

# First load the HF model with quantization
hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=quantization_config,
    cache_dir="./models/"
)

# Then pass it to transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    hf_model=hf_model,
    cache_dir="./models/"
)
print(model)
