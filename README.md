# Pyramid_Based_Texture_Synthesis
Python implementation of the Heeger–Bergen pyramid-based texture synthesis algorithm using steerable pyramids and histogram matching.
Medium (recommended GitHub description)
This repository contains a Python implementation of the Heeger–Bergen texture synthesis algorithm, originally introduced in Pyramid-based Texture Analysis/Synthesis.
The method synthesizes textures by iteratively matching first-order statistics in both the image domain and a multiscale, multi-orientation steerable pyramid representation.
Starting from white noise, the algorithm alternates between:
•	Histogram matching in the pixel domain 
•	Histogram matching across pyramid subbands 
until convergence, producing textures that are perceptually similar to the input but structurally different.
This implementation is inspired by the detailed IPOL article:
The Heeger-Bergen Pyramid-Based Texture Synthesis Algorithm
Features
•	Grayscale and (optional) color texture synthesis (via PCA) 
•	Steerable pyramid decomposition and reconstruction 
•	Exact histogram matching 
•	Configurable parameters (scales, orientations, iterations) 
Applications
•	Texture synthesis and generation 
•	Image processing research 
•	Computer graphics and procedural content

