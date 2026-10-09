# Survey on Video and Multimodal Generation

## Executive Summary (TL;DR)
Video generation is undergoing a paradigm shift from traditional pixel-space diffusion to latent-space Diffusion Transformers (DiTs). Current state-of-the-art models like Kandinsky 6.0 Video [4] emphasize synchronized audio-video synthesis. Modern architectures prioritize long-form coherence via techniques like FrameMorrow [5] and hybrid gradient forcing (SGF+) [6], while foundational research investigates advanced benchmarks like the World Consistency Score (WCS) [2] and WorldJen [1] to address the limitations of existing video quality metrics.

## Background and Problem Formulation
Traditional generative video models historically relied on pixel-level diffusion, which proved computationally prohibitive for high-resolution, long-form content. As noted in [8], moving generation into compressed latent spaces via Video-VAEs allows for more scalable spatio-temporal modeling. The core challenge in video generation is no longer just visual quality, but ensuring semantic consistency and long-horizon temporal stability, a problem currently addressed by moving beyond simple text-conditioning towards MLLM-DiT fusion architectures [3].

## Core Architectures and Methodologies
The current architectural consensus centers on the Diffusion Transformer (DiT). Pioneering work like Latte [8] demonstrated that Transformer blocks can effectively capture latent video distributions. To enable synchronized multimodal outputs, models like UniForm [9] utilize unified latent spaces where audio and video are generated simultaneously. For resource-constrained environments, approaches like Neodragon [7] utilize distillation and compression to enable high-quality video generation on mobile devices.

## Recent Advances and State-of-the-Art
Recent trends focus on improving long-horizon generation and semantic control. Kandinsky 6.0 Video [4] represents a significant leap in T2AV (Text-to-Audio-Video) synthesis using a dual-stream CrossDiT architecture. Methods like FrameMorrow [5] introduce prospective frame selection to improve future-sequence coherence, while SGF+ [6] improves temporal consistency by decoupling gradient flows for content and denoising processes.

## Trends, Open Challenges, and Future Directions
The field is increasingly moving toward "World-Action Models" that integrate reasoning, world generation, and action prediction [4]. Evaluation remains a primary challenge; conventional metrics like FVD are increasingly viewed as insufficient [1]. Emerging standards like the World Consistency Score (WCS) [2] focus on object permanence and causal compliance, providing a more robust framework for evaluating model performance compared to legacy metrics. Future efforts will likely focus on unified world models that provide both high-fidelity video generation and reliable physical reasoning.

## References
[1] WorldJen: An End-to-End Multi-Dimensional Benchmark for Generative Video Models. https://arxiv.org/abs/2605.03475 (arxiv)
[2] World Consistency Score: A Unified Metric for Video Generation Quality. https://arxiv.org/abs/2508.00144 (arxiv)
[3] Beyond Text Conditioning: A Systematic Study of MLLM-DiT Fusion for Video Generation. https://arxiv.org/abs/2608.14043 (arxiv)
[4] Kandinsky 6.0 Video. https://huggingface.co/papers/2610.05608 (hf-daily)
[5] FrameMorrow: Future-guided Frame Selection. https://huggingface.co/papers/2609.38839 (hf-daily)
[6] SGF+ (Self Gradient Forcing Plus). https://huggingface.co/papers/2610.10429 (hf-daily)
[7] Neodragon: Mobile Video Generation using Diffusion Transformer. https://huggingface.co/papers/2511.06055 (hf-search)
[8] Latte: Latent Diffusion Transformer for Video Generation. https://huggingface.co/papers/2401.03048 (hf-search)
[9] UniForm: A Unified Diffusion Transformer for Audio-Video Generation. https://huggingface.co/papers/2502.03897 (hf-search)
