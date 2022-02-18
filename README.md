# raw_stack

Stack fits (astro-camera) or raw (DSLR) format astro-images with: 

1) Automatic alignment.
2) Absolute rotation correction based on the time, site location, and observation target. 
3) Automatic selection of the reference frame based on cross-covariance with other frames. 
4) Automatic computation of the weights.
5) Automatic refraction correction (requires color-image and working in color-mode).
6) No explicit dependence on the point sources, but one still needs to adjust some alignment parameters to ensure a correct alignment, especially when the SNR is low.

