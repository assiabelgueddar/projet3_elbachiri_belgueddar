import torch
from diffusers import DiffusionPipeline
import gradio as gr

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float32
).to("cpu")

pipe.load_lora_weights(
    "./trained-sd3-lora-miniature",
    weight_name="adapter_model.safetensors"
)

def generate_image(prompt, progress=gr.Progress()):
    progress(0, desc="Chargement...")
    image = pipe(prompt=prompt, height=384, width=384, num_inference_steps=10).images[0]
    return image

with gr.Blocks(css="""
#main {
    max-width: 800px;
    margin: auto;
    font-family: 'Segoe UI', sans-serif;
}
.gr-button.primary,.svelte-1ixn6qd {
    background: linear-gradient(to right, #1974dc, #5faaff) !important;
    color: white;
    border-radius: 8px;
    font-size: 1.1rem;
    padding: 0.6rem 1.5rem;
}
""") as demo:
    with gr.Column(elem_id="main"):
      
        gr.Image(value="logo.png", show_label=False, container=False)
        gr.HTML("""
        <div style="text-align: center; padding: 1em;">
           
            <h1 style="font-size: 2.5em;">Text to Image Projet</h1>
           
        </div>
        """)

        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="a photo of cat in a bucket",
            show_label=True
        )
        generate_button = gr.Button("Generate Image", elem_classes=["primary"])
        output_image = gr.Image(label="Generated Image", type="pil")
        generate_button.click(fn=generate_image, inputs=prompt_input, outputs=output_image)
demo.launch()
