## Segment Anything (SAM) Dependency

Some experiments in this repository rely on the Segment Anything Model (SAM) by Meta AI.  
To successfully run SAM-related experiments, users must install the SAM environment and download the required pretrained checkpoints.

Please follow the official SAM installation instructions and model download steps:

- SAM GitHub Repository (Installation & Usage):  
  https://github.com/facebookresearch/segment-anything

- Pretrained Checkpoints:  
  https://github.com/facebookresearch/segment-anything#model-checkpoints

Ensure that:
- The Python environment and dependencies match the SAM requirements.
- The downloaded checkpoint files are placed in the correct directory as expected by the SAM code.
- CUDA-enabled PyTorch is installed if GPU acceleration is desired.

Failure to complete these steps will prevent SAM-based components of this repository from running correctly.