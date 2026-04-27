# LivePortrait Expression Swap — Free Colab Notebook

Transfer facial expressions between photos using [LivePortrait](https://github.com/KwaiVGI/LivePortrait). Copy a smile, a laugh, a surprised look — any expression from one face onto another, while preserving the target person's identity.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/face2faceai/liveportrait-expression-swap/blob/main/liveportrait_expression_swap_colab.ipynb)    

---

## What It Does

Load two face photos — a **source** (the expression you want) and a **target** (the face you want to change). The notebook extracts the expression from the source and applies it to the target, producing a new image where the target person now wears the source person's expression.

The target's identity, face shape, skin tone, and hair are preserved. Only the expression transfers.

### Example

<table>
  <tr>
    <td align="center"><img src="https://face2faceai.com/images/expr-before.png" width="220" alt="Target face — before"><br><sub><b>Your photo</b></sub></td>
    <td align="center"><b>&nbsp;&nbsp;+&nbsp;&nbsp;</b></td>
    <td align="center"><img src="https://face2faceai.com/images/expr-template.png" width="220" alt="Zara Silly 2 expression template"><br><sub><b>Zara · Silly 2 template</b></sub></td>
    <td align="center"><b>&nbsp;&nbsp;=&nbsp;&nbsp;</b></td>
    <td align="center"><img src="https://face2faceai.com/images/expr-after.png" width="220" alt="Result after expression transfer"><br><sub><b>Result</b></sub></td>
  </tr>
</table>

> **Note:** The result image shows face reinsertion — the blended face placed back into the original photo. Face reinsertion is a feature of the Face2FaceAI app; the notebook outputs a 512×512 cropped face.

---

## Quick Start

1. Click **Open in Google Colab** above
2. Click **Runtime → Run all** — there is only one cell to run
3. First run takes 4–5 minutes to install dependencies and download model weights (~560 MB, cached for future sessions)
4. A Gradio link appears at the bottom of the cell output — click it to open the interface in your browser
5. In the Gradio interface, upload your source image in the **Source Image** window and your target image in the **Target Image** window
6. Adjust blend sliders to control how strongly the expression transfers
7. Download or save your result

**Requirements:** A Google account and a free Colab session with GPU runtime. Before running: select **Runtime → Change runtime type → T4 GPU**. CPU-only runtime is not supported.

---

## Features

- **Expression transfer** — Copy any facial expression from one photo to another
- **Blend control** — Adjust the strength of expression transfer with a slider from subtle to exaggerated
- **Head rotation blending** — Independently control how much head pose transfers
- **Automatic face crop** — Automatic face detection in a single-face image
- **Output mode** — Export as a 512×512 upsampled face
- **Works with any face photo** — Selfies, portraits, or photorealistic AI-generated faces

The Face2FaceAI app adds adjustable face cropping, support for multi-face images, asymmetry correction via mirroring, and additional output modes, including face reinsertion.

---

## How It Works

This notebook uses the [LivePortrait](https://github.com/KwaiVGI/LivePortrait) pipeline from KwaiVGI, which decomposes facial motion into appearance and expression components. The key insight is that expressions can be separated from identity — the system encodes "what your face is doing" independently from "who you are," making it possible to transfer one to the other.

Under the hood: face detection (MediaPipe) → expression encoding → latent space blending → image generation. All processing runs locally in your Colab session — no images are uploaded to any external service.

---

## Use Cases

- **Just for fun** — Give yourself a different expression in a selfie, make friends laugh
- **Content creation** — Create expression variations for social media, stickers, or memes
- **Selfie correction** — Fix an awkward smile or closed eyes in an otherwise good photo
- **Character design** — Apply a range of expressions to a photorealistic AI-generated face

---

## Limitations

- Works best with clear, front-facing photos where the face is well-lit
- Extreme head angles (profile views) may produce artifacts
- Very different face shapes between source and target can cause distortion
- Processing time is ~2–8 seconds per image on a T4 GPU

---

## Privacy

All processing runs locally in your Google Colab session. No face images are uploaded to any external server. When your Colab session ends, all uploaded images and generated results are deleted.

---

## Face2FaceAI — The App Version

This notebook demonstrates the core concept of expression swapping. **Face2FaceAI** is a full Android app built on the same underlying technology, optimized for mobile with on-device ONNX Runtime inference — no internet needed after initial setup.

The app goes significantly beyond what this notebook offers, with advanced face selection and blending controls, a curated template library of ready-made expressions, multiple output modes, and quality refinements not available here. All processing stays entirely on your phone.

**[face2faceai.com](https://face2faceai.com)**

<!-- Uncomment when live:
[**Download on Google Play**](PLAY_STORE_LINK_HERE)
-->

---

## Support This Project

If this notebook is useful to you, consider supporting ongoing development:

<!-- Uncomment and add your links:
- [GitHub Sponsors](GITHUB_SPONSORS_LINK)
- [Buy Me a Coffee](BUYMEACOFFEE_LINK)
-->

---

## Credits & License

- **LivePortrait** by [KwaiVGI](https://github.com/KwaiVGI/LivePortrait) — MIT License
- **MediaPipe** by Google — Apache 2.0 (replaces InsightFace for commercial compatibility)

This notebook is released under the MIT License.

---

## Questions or Feedback?

Open an issue on this repository, or reach out at **[dev@face2faceai.com](mailto:dev@face2faceai.com)**.
