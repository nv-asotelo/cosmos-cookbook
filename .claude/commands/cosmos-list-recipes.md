List all available Cosmos Cookbook recipes organized by category and model.

Steps:
1. Read /f/git/cosmos-cookbook/docs/recipes/all_recipes.md for the canonical recipe list.
2. Scan the directory tree under docs/recipes/ to find all directories containing a CLAUDE.md. This catches any recipes added after all_recipes.md was last updated.
3. Display recipes in four groups:

**Inference Recipes** (grouped by model)
- Cosmos Predict 2: ITS Image Synthesis
- Cosmos Reason 2: Intbot Edge VLM, Intbot Showcase, Vector Search System, Worker Safety
- Cosmos Transfer 1: GR00T Mimic, ITS Weather Augmentation, Warehouse Movement, X-Mobility
- Cosmos Transfer 2.5: Biotrove Augmentation, CARLA SDG, Image Prompt, Real-World Augmentation

**Post-Training Recipes** (grouped by model)
- Cosmos Predict 2: Cosmos Policy, GR00T Dreams, ITS Accident
- Cosmos Predict 2.5: Sports, Surgical Robotics
- Cosmos Reason 1: AV Caption & VQA, Intelligent Transportation, Physical Plausibility, Spatial AI Warehouse, Temporal Localization, Wafermap Classification
- Cosmos Reason 2: AV 3D Grounding, Intelligent Transportation, Physical Plausibility, Video Caption & VQA
- Cosmos Transfer 2.5: AV World Scenario Maps

**Data Curation**
- CABR Video Compression, Embedding Analysis, Outlier Detection, Predict 2 Data Curation

**End-to-End Workflows**
- GR00T Dreams (Predict 2.5 + Reason 2), Smart City SDG (Transfer 2.5 + Reason 1 + CARLA)

4. For each recipe, show: name, directory path, and domain (Robotics / Autonomous Vehicles / Vision AI / Industrial / Medical / Life Sciences).
5. Show total count and suggest: "Run /cosmos-run-recipe <recipe-name> to execute any recipe. Run /cosmos-setup to validate your environment first."
