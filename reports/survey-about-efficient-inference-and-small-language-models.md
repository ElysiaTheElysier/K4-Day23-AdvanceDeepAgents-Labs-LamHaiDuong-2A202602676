# Survey on Efficient Inference and Small Language Models

## Executive Summary
Recent advancements in Small Language Models (SLMs) and efficient inference have shifted from simple parameter reduction to holistic optimization. This survey explores methods including Mixture-of-Experts (MoE) pruning, phase-aware serving, and hardware-software co-design. We examine academic foundations [1][2][3], recent trending techniques [4][5][6], and specialized architectures [7][8][9] to highlight the path toward deployable and performant edge language models.

## Background and Problem Formulation
Large Language Models (LLMs) often face significant deployment challenges due to massive memory footprints and latency constraints. Small Language Models (SLMs) have emerged as a critical research area to enable on-device or edge deployment [3]. The foundational challenge involves maintaining the reasoning capabilities of large models while drastically reducing their compute and memory requirements through techniques like pruning, quantization, and architectural optimization [1][2].

## Core Architectures and Methodologies
Efficient inference in modern language models often leverages MoE architectures. Research has focused on expert-level pruning, where redundant or low-impact experts are removed based on router hints [8]. Techniques like `MC-MoE` provide training-free compression using mixed-precision quantization [9]. For general transformer models, hardware-aware pipelines—integrating pruning, knowledge distillation, and quantization—have proven effective for CPU-based inference [7].

## Recent Advances and State-of-the-Art
Current trends show a move toward specialized inference pipelines. `SlimWise` proposes a strategy to decouple pruning between prefill and decode stages, optimizing serving throughput [6]. In terms of reinforcement learning (RL) for MoE models, `TRACE` enables quantization-aware training in FP4 precision to minimize error propagation [5]. Furthermore, research into linear attention models, such as `STEPQuant`, highlights the importance of temporal-spatial error analysis when quantizing recurrent states [4].

## Trends, Open Challenges, and Future Directions
The field is increasingly moving toward interpretability-driven efficiency, where probing analysis informs subnetwork selection [1]. Additionally, the introduction of "Parallel Scaling" suggests that scaling parallel compute capacity, rather than just depth or parameter count, is a viable efficiency path [2]. Future directions include the continued hardware-co-design of models to bridge the gap between abstract algorithmic efficiency and physical device constraints [3].

## References
[1] Fast DistilBERT on CPUs. hf-search. https://huggingface.co/papers/2211.07715 (2022-10-27)
[2] MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router. hf-search. https://huggingface.co/papers/2410.12013 (2024-10-15)
[3] MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More. hf-search. https://huggingface.co/papers/2410.06270 (2024-10-08)
[4] STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization. hf-daily. https://huggingface.co/papers/2609.38169 (2026-09-29)
[5] TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models. hf-daily. https://huggingface.co/papers/2610.07767 (2026-10-06)
[6] SlimWise: Decoupling Expert Pruning Across Prefill and Decode for Efficient MoE Serving. hf-daily. https://huggingface.co/papers/2609.34117 (2026-09-28)
[7] ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference. arxiv. https://arxiv.org/abs/2606.01806 (2026-06-01)
[8] Parallel Scaling Law for Language Models. arxiv. https://arxiv.org/abs/2505.10475 (2025-05-15)
[9] PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing. arxiv. https://arxiv.org/abs/2503.12167 (2025-03-15)
