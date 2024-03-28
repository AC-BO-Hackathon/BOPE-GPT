# bope-gpt

To run the code, I'm typically updating a conda/mamba environment that, on the first time, can be installed using the following commands:

`mamba create -n botorch_mar2024 pytorch torchvision torchaudio pytorch-cuda=11.8 python==3.11 -c pytorch -c nvidia`

`mamba install botorch matplotlib seaborn -c pytorch -c gpytorch -c conda-forge`

`mamba update -c conda-forge ffmpeg`

`mamba install -c conda-forge dash`

**The first steps**

*Analysing the Fischer-Tropsch dataset from the point of view of classical single and multi-objective B0*

Fischer-Tropsch Synthesis represents a pivotal process in the field of industrial chemistry, serving as a cornerstone for the production of liquid hydrocarbons from carbon monoxide and hydrogen gases. Developed by German chemists Franz Fischer and Hans Tropsch in the early 1920s, this method provides a versatile pathway for converting syngas—a mixture of hydrogen and carbon monoxide derived from coal, biomass, or natural gas—into a variety of valuable hydrocarbon products, including fuels and alkanes. The process is particularly adopted for its ability to produce clean, sulfur-free fuels, which are crucial in today's efforts towards environmental sustainability and energy security. Through catalytic chemical reactions conducted at high temperatures and pressures, Fischer-Tropsch Synthesis offers a strategic approach to mitigating reliance on crude oil by leveraging alternative carbon sources, thereby playing a critical role in the evolving landscape of global energy.

The Fischer-Tropsch synthesis is a chemical reaction that converts a mixture of carbon monoxide (CO) and hydrogen gas (H₂) into liquid hydrocarbons.

\[ n\,CO + (2n+1)\,H_2 \rightarrow C_nH_{2n+2} + n\,H_2O \]


**Into the preference world**

**An app to rule them all**
