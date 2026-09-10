# robo-diffusion-sampling-method

## Status
🚧 Work in Progress

Core structure created:
- ✅ task_description.md
- ✅ config.json
- ✅ parser.py
- ✅ Training scripts
- ⏳ Template and baselines (to be implemented)

## Baselines
- ddpm: DDPM sampling with 100 steps - Standard but slow
- ddim: DDIM sampling with 20 steps - Faster deterministic sampling
- dpm_solver: DPM-Solver++ with 10 steps - Fastest high-quality sampling

## Next Steps
1. Create custom_template.py with editable region
2. Create mid_edit.py
3. Implement baseline edit files
4. Test with CleanDiffuser
