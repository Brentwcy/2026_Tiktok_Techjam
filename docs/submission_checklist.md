# Final submission checklist

## Completed

- [x] Public GitHub repository
- [x] EfficientNet-B0 training pipeline
- [x] Exact Track 5 training augmentations
- [x] Clean and per-transform robustness evaluation
- [x] Error summary and training log committed under `results/`
- [x] Sanitized representative false-positive/false-negative examples
- [x] Raw DINOv2/DINOv3 comparison metrics committed under `results/`
- [x] Required directory-to-JSON inference script
- [x] Public `best.pt` checkpoint
- [x] Fresh-clone checkpoint download and inference test
- [x] CPU/GPU Streamlit and Gradio demos (including live Track 5 transformation toggle and side-by-side comparison)
- [x] Devpost description draft
- [x] Demo video script with live robustness demonstration walkthrough

## Must complete before submission

- [ ] Replace the README and Devpost team section with real names and actual contributions
- [ ] Record the demo video using `docs/demo_script.md`
- [ ] Upload the video publicly/unlisted as allowed by the rules
- [ ] Add the final video URL to Devpost
- [ ] Copy `docs/devpost_description.md` into the Devpost fields
- [ ] Add the repository and checkpoint links to Devpost
- [ ] Complete the event registration/submission form
- [ ] Open the repository, checkpoint, video, and Devpost links in an incognito window
- [ ] Submit at least 30–60 minutes before the deadline

## Optional only if all required items are finished

- [ ] Evaluate on the reserved WildFake/COCO–DALL·E benchmark
- [ ] Add SID_Set for generator and resolution diversity
- [ ] Deploy the Gradio app to persistent hosting instead of a temporary share link
