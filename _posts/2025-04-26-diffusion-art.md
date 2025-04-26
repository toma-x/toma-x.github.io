---
layout: post
title: Building a Diffusion Model for Art Generation
---

Hey everyone,

After spending some time on LeetCode and then exploring multimodal models like VLLMs and VQA agents, I wanted to tackle something that felt a bit more... creative. I've been blown away by AI art generation models like Stable Diffusion, DALL-E 2, and Midjourney, and I really wanted to understand how they work under the hood. So, for my latest project, I decided to try and build my own text-to-image diffusion model pipeline, heavily inspired by Stable Diffusion. My main goal was to get hands-on experience with the core components and experiment with using CLIP embeddings to control the generation process.

## What's the Deal with Diffusion Models?

Before diving in, I had to wrap my head around diffusion models. The core idea is surprisingly elegant, even if the math gets hairy fast. It works in two main stages:

1.  **Forward Process (Adding Noise):** You start with a real image and gradually add a little bit of random noise (Gaussian noise) over many, many steps. Eventually, after enough steps, the original image just looks like pure static. This part is easy – just adding noise.
2.  **Reverse Process (Denoising):** This is the magic part. You train a neural network to *reverse* the process. Given a noisy image at a certain step, the model learns to predict the noise that was added to get there (or equivalently, predict the slightly less noisy image). If you can do this repeatedly, starting from pure random noise, you can gradually "denoise" your way back to a clean image.

Crucially, for text-to-image, the model doesn't just denoise randomly; it's *guided* by the text prompt.

## My Approach: Following the Stable Diffusion Blueprint

Stable Diffusion is incredibly complex, so building an exact replica from scratch was out of scope. However, I focused on understanding and implementing the key components, leveraging pre-trained parts where necessary (which is how most people interact with these models anyway!). I primarily used PyTorch and the Hugging Face `diffusers` library, which was an absolute lifesaver for providing building blocks and pre-trained weights.

Here are the main pieces I worked with:

1.  **Variational Autoencoder (VAE):** Stable Diffusion doesn't work directly on pixel space. That would be computationally massive. Instead, it uses a VAE to encode the image into a smaller *latent space*. The diffusion process (adding and removing noise) happens in this compressed space, which is much more efficient. After the denoising is done in latent space, a VAE decoder converts the final latent representation back into a visible image. I used a pre-trained VAE compatible with Stable Diffusion.

2.  **Text Encoder (CLIP):** How does the model understand the text prompt like "a cat riding a bicycle"? This is where CLIP (Contrastive Language–Image Pre-training) comes in. CLIP is trained on millions of image-text pairs and learns to map both text descriptions and images into a shared embedding space. We use the pre-trained CLIP text encoder to convert the input prompt into numerical embeddings (vectors). These embeddings capture the *meaning* of the text. I used the standard pre-trained CLIP ViT-L/14 text encoder.

3.  **U-Net Denoising Model:** This is the core of the reverse process. It's a neural network (typically with a U-Net architecture – an encoder-decoder structure with skip connections) that takes the noisy latent image from the previous step *and* the CLIP text embeddings as input. Its job is to predict the noise present in the latent image at that specific timestep. I used a pre-trained U-Net model compatible with Stable Diffusion v1.5.

4.  **Scheduler:** The scheduler defines the noise schedule – how much noise is added in the forward process and how the timesteps are used during the reverse (denoising) process. It calculates the variance of noise for each step. Different schedulers can affect the speed and quality of generation. I experimented with a few common ones available in `diffusers`, like PNDM, DDIM, and Euler Ancestral.

## Putting it Together: The Generation Pipeline

Here’s a simplified overview of how the generation process (sampling) works, conceptually similar to what the `diffusers` library helps manage:

```python
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler # Example Scheduler
# Assuming models are loaded and on the correct device (e.g., GPU)

# 1. Settings
prompt = ["a photograph of an astronaut riding a horse"]
height = 512
width = 512
num_inference_steps = 50 # Number of denoising steps
guidance_scale = 7.5 # Classifier-Free Guidance scale
generator = torch.Generator("cuda").manual_seed(0) # For reproducibility

# 2. Get Text Embeddings using CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]

# Handle Classifier-Free Guidance (using unconditional embeddings)
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt") # Empty prompt for unconditional guidance
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) # Combine for CFG

# 3. Prepare Scheduler and Initial Noise
scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler") # Example
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to("cuda")
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to("cuda")

latents = torch.randn(
    (1, unet.in_channels, height // 8, width // 8), # Initial random noise in latent space (height/width divided by VAE scale factor)
    generator=generator,
    device="cuda",
)
latents = latents * scheduler.init_noise_sigma # Scale noise appropriately

# 4. Denoising Loop
scheduler.set_timesteps(num_inference_steps)
for t in scheduler.timesteps: # Iterate backwards through timesteps
    # Expand latents for CFG (duplicate for conditional and unconditional)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t) # Scale input if scheduler requires

    # Predict noise using U-Net
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform Classifier-Free Guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute previous noisy sample (step backwards)
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# 5. Decode Latent to Image using VAE
latents = 1 / 0.18215 * latents # Magic scaling factor used in Stable Diffusion
with torch.no_grad():
    image = vae.decode(latents).sample

# 6. Post-process image (convert to PIL, etc.)
# ... image processing code ...

```
*(Disclaimer: This is simplified. The actual `diffusers` pipeline handles more details and edge cases!)*

## Playing with CLIP Embeddings and Guidance

This was the most fun part. CLIP embeddings are the bridge between text and the U-Net's noise prediction.
*   **Prompt Engineering:** Changing even small words in the prompt drastically alters the CLIP embeddings and thus the final image. "A *photo* of an astronaut..." looks different from "A *painting* of an astronaut...". Adding style keywords like "cinematic lighting," "vaporwave," or "drawn by Van Gogh" really steered the output.
*   **Classifier-Free Guidance (CFG):** That `guidance_scale` parameter is super important. It controls how strongly the model should adhere to the text prompt versus just generating a "typical" image (unconditional generation). Higher values (like 7-12) make the image follow the prompt more closely, but too high can lead to weird artifacts or oversaturated colors. Lower values give the model more creative freedom. Finding the sweet spot was key.
*   **Negative Prompts:** The CFG trick relies on having embeddings for your actual prompt and *unconditional* embeddings (usually from an empty string). I experimented with using embeddings from *negative* prompts here instead of the empty string. For example, if I generated "a beautiful landscape" but got lots of buildings, I could use "buildings, structures" as a negative prompt to steer the generation *away* from those concepts. This worked surprisingly well for removing unwanted elements or fixing common failure modes (like extra fingers).

## Challenges: The Struggle is Real

*   **GPU Memory:** Seriously, this is the biggest hurdle. Even using pre-trained models, the U-Net, VAE, and CLIP models are *huge*. Loading them all onto a GPU requires significant VRAM (often 10GB+ just for inference, much more for training/fine-tuning). I relied heavily on Google Colab Pro with high-RAM GPUs. Even then, generating larger images (like 768x768) was pushing it. Forget fine-tuning the U-Net on my local machine!
*   **Complexity Overload:** While `diffusers` abstracts a lot, understanding *why* certain things work requires diving into the code and sometimes the original papers. Concepts like schedulers, CFG, and the VAE latent space took time and effort to grasp properly. Debugging when things went wrong often involved tracing steps through the pipeline.
*   **Tuning Knobs:** Finding the right combination of scheduler, number of steps, and guidance scale for a given prompt felt like an art form. Too few steps, and the image is noisy/unfinished. Too many steps, and it takes forever (and doesn't always improve much). Different schedulers also have different speed/quality trade-offs. Lots of trial and error!
*   **Prompt Isn't Everything:** Sometimes, no matter how I tweaked the prompt, the model just wouldn't generate what I wanted, or it would produce strange artifacts (the classic AI hands problem!). Diffusion models aren't perfect reasoning engines; they are predicting noise based on patterns learned from data.

## Some Results (Descriptions)

I managed to generate some pretty cool images!
*   "A watercolor painting of a robot sitting on a park bench": Got a nice, soft image with painterly textures and a recognizable robot.
*   "A high-detail photograph of a cat wearing sunglasses, cinematic lighting": Produced a sharp image with good lighting, capturing the prompt elements well.
*   "An astronaut riding a horse on the moon": This classic prompt worked surprisingly well, generating a coherent scene.

Of course, there were failures:
*   Requests for specific text often failed – the model might generate something that *looks* like text but is just gibberish.
*   Complex scenes with many interacting objects sometimes resulted in weird spatial relationships or blended objects.
*   Counting specific numbers of objects was hit-or-miss.

## What I Learned

This project was incredibly insightful:
*   **Diffusion Models Demystified:** I moved from "it's magic" to having a practical understanding of the denoising process, the role of the U-Net, VAE, and scheduler.
*   **CLIP is Powerful:** I gained a huge appreciation for how CLIP embeddings enable text conditioning and stylistic control. Experimenting with prompts and negative prompts really drove this home.
*   **Engineering Challenges:** I got first-hand experience with the computational demands of large generative models and the importance of libraries like `diffusers` for making them accessible.
*   **Iteration is Key:** AI development, especially with generative models, involves a lot of experimentation, parameter tuning, and learning from failures.

## Conclusion

Building (or rather, assembling and experimenting with) this Stable Diffusion-inspired pipeline was challenging but super rewarding. It deepened my understanding of modern generative AI and gave me practical experience with the tools and techniques involved. While I mostly used pre-trained components, manipulating the pipeline, especially through prompt engineering and guidance techniques using CLIP embeddings, felt like I was genuinely steering the creative process. It's an amazing field, and I'm excited to see how these models continue to evolve!

Happy to discuss this more if you're curious!