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

$$ n CO + (2n+1) H_2 \rightarrow C_nH_{2n+2} + n H_2O $$

The ground truth we use here is the Artifitial Neural Network model built from the dataset in the paper (Chakkingal, Anoop, et al., 2022), with four inputs: space-time, syngas ratio, temperature and pressure, and four outputs: carbon monoxide conversion, and the selectivity towards methane (SCH4), paraffins (SC2−C4) and light olefins (SC2−C4=).

We set the objective function as maximizing all of the four outputs. However, if in the reality some of the three products are considered as byproducts, we can adjust the objective settings in BO and make the optimization problem more adapted to the reality.

*Single-objective BO implementation*

*Multi-objective BO implementation*
We then implemented multi-objective BO to explore the pareto front in the Fischer-Tropsch dataset and identify poteintial trade-offs. We keep the SingleTaskGP as the model, and the qExpectedHypervolumeImprovement as the acquisition function to see the pareto front. By looking at the pairwise comparision of the output pairs, we found for some pairs of the output, the trade-off is quite clear (e.g. output 1 and output 3). However, some of them are hard to identify (e.g. output 3 and output 4).

![image](https://github.com/AC-BO-Hackathon/BOPE-GPT/assets/113897191/c28430e4-b81d-413b-9fe5-3e016a1bcc53)
![image](https://github.com/AC-BO-Hackathon/BOPE-GPT/assets/113897191/d7b97464-eb85-4e70-8ad0-5f28516559de)



**Into the preference world**
From the multi-object BO, we saw that the multi-objective optimization result could be a very hard task using the distance to the pareto front, because it is very hard to define the objective. Therefrore, we introduced a preference setting to the Fischer-Tropsch problem, and expect the LLM to do the pairwise comparison.

*Decision by a comparison function*
We first used a comparison function to 

*Decision by an LLM*
Finally we turned to the pairwise comparison by LLM.

**An app to rule them all**
