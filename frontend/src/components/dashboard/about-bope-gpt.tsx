import React from 'react';
import { 
  Accordion, 
  AccordionContent, 
  AccordionItem, 
  AccordionTrigger 
} from "@/components/ui/accordion";
import { Separator } from '@radix-ui/react-separator';

export function BOPEFAQAccordion() {
  return (
        <div className="w-full max-w-4xl mx-auto p-4">
            <h2 className="text-2xl font-semibold mb-4">BOPE-GPT FAQ</h2>
            <Separator orientation="horizontal" className="mb-4" />
        <Accordion type="single" collapsible>
            <AccordionItem value="what-is-bope">
            <AccordionTrigger>What is the BOPE process?</AccordionTrigger>
            <AccordionContent>
                <p>
                BOPE, or <a href="https://botorch.org/tutorials/bope" target="_blank" rel="noopener noreferrer" className="text-blue-500">Bayesian Optimization via Preference Exploration</a>, is a machine learning technique for finding potential optimums of an experiment/system (or in more abstract terms, any complex or unknown function that maps some inputs to some outputs) using pairwise comparisons of data points sampled from the distribution of potential inputs to guide this optimization process.
                </p>
                <br />
                <p>
                Typically, bayesian optimization of a function requires numerically defining what optimum output should be strived towards finding. However, in multi-objective problems, it is often the case that specific priorities - the optimization goals - <em>cannot</em> be defined well numerically. 
                </p>
                <br />
                <p>
                Using pairwise comparisons between outputs is useful for such situations. In these scenarios, a human (or human-like) evaluator/decision-maker can express a &quot;preference&quot; between two outputs and this preference can be used to guide the optimization process instead.
                </p>
                <br />
                <p>
                This &quot;preference&quot; can also be picked through an LLM (the human-like evaluator) if correctly prompted - which is what BOPE-GPT does.
                </p>
                <br />
                <p>
                After each iteration, the list of explored data points expands - Pareto plot and preference visualizations of the pairwiseGP model representing these can then be used to locate optimal points in the distribution by a user.
                </p>
                <br />
            </AccordionContent>
            </AccordionItem>

            <AccordionItem value="whos-this-for">
            <AccordionTrigger>Who&apos;s this web app for?</AccordionTrigger>
            <AccordionContent>
                <p>
                This is for people who want to optimize a multi-objective system they&apos;re trying to model and already have a distribution of input-output data in a dataset, but don&apos;t have a good way to define the optimization goals numerically - although they can in natural language.
                </p>
                <br />
                <p>
                In the future, support will be added for using this with live experiments, requiring new data points to be entered once prompted after the model makes a pair of suggestions in the course of each iteration. 
                </p>
                <br />
                <p>
                The BOPE-GPT process is a useful interface for this kind of optimization, with built-in visualizations to help see the latent utility defined in natural language and select potential optimums. 
                </p>
                <br />
                <p>
                    With an LLM and the interpolation abilities of a sufficiently trained neural network, this app also automates some of the time consuming requirements of the BOPE process -along with potential human biases a human preference selector may unconsciously introduce- 
                    especially for repetitive but similar runs. Anything such task where the objectives can be defined in natural language (to be prompted to the LLM) is also useful to perform with BOPE-GPT. 
                </p>
                <br />
                <div>
                <p>Potential use cases include:</p>
                <ul className="list-disc pl-6 mt-2">
                    <li>Chemical processes with tradeoffs in outputs (e.g., Fischer-Tropsch synthesis process)</li>
                    <li>A/B tests (as mentioned in the BOPE paper)</li> 
                    <li>Simulation-based design (as mentioned in the BOPE paper)</li>
                    <li>Tuning hyperparameters of other ML models</li>
                    <li>Optimization tasks where objectives cannot be defined easily numerically but can be explained in natural language</li>
                    <li>Any multi-objective task where pairwise preferences are a good way of making comparisons between outputs</li>
                </ul>
                </div>
            </AccordionContent>
            </AccordionItem>

            <AccordionItem value="how-does-bope-gpt-work">
            <AccordionTrigger>How does BOPE-GPT work?</AccordionTrigger>
            <AccordionContent>
            <ul className="list-disc pl-6 space-y-2">
                <li>BOPE-GPT first awaits an upload of an initial CSV dataset - tabular, with columns representing features and rows representing individual data points</li>
                <li>To initialize the BOPE Process, the Initialization Panel must be filled out:
                    <ul className="list-disc pl-6">
                    <li>An LLM prompt</li>
                    <li>Number of dataset columns to use as inputs (outputs defined by elimination)</li>
                    <li>Number of initial data points to sample from the dataset distribution</li>
                    <li>Number of initial pairwise comparisons to make</li>
                    </ul>
                </li>
                <li>A neural network is created to internally represent the distribution of the uploaded dataset, mapping inputs to outputs</li>
                <li>The BOPE process begins, iteratively sampling new data points from the distribution and comparing them using a PairwiseGP model</li>
                <li>An LLM is used to automate the pairwise comparisons between data points</li>
                <li>Visualizations of the current state of the BOPE process are updated as it runs:
                    <ul className="list-disc pl-6">
                    <li>PairwiseGP model</li>
                    <li>Pareto Fronts of explored data points</li>
                    </ul>
                </li>
                <li>Overviews, visualizations, data, and comparisons can be viewed via dashboard tabs after each iteration</li>
                <li>Each iteration improves the PairwiseGP model and increases the likelihood of finding more optimal data points</li>
                <li>To identify desired optimization solutions, the visualizations and current model data tabs can be examined after a sufficient number of model iterations have been run</li>
                </ul>
            </AccordionContent>
            </AccordionItem>

            <AccordionItem value="example-initialization">
            <AccordionTrigger>How can I try this out? (Example BOPE-GPT Initialization Panel Values)</AccordionTrigger>
            <AccordionContent>
            <div className="space-y-4">
                <div>
                    <h3 className="font-semibold">Sample dataset:</h3>
                    <p>Download <a className="text-blue-500" href="https://github.com/AC-BO-Hackathon/BOPE-GPT/blob/main/data/fischer_data_processed.csv" target="_blank" rel="noopener noreferrer">this Fischer-Tropsch dataset</a> and upload it onto this app to enable the initialization panel</p>
                </div>
                <div>
                    <h3 className="font-semibold">LLM Prompt:</h3>
                    <p className="italic">
                    &quot;Suppose you&apos;re managing a Fischer-Tropsch synthesis process. The four outputs are equally important, and we want to maximize all of them&quot;
                    </p>
                </div>
                <div>
                    <h3 className="font-semibold">Number of Input Features:</h3>
                    <p>4 (the first four columns of the uploaded dataset)</p>
                </div>
                <div>
                    <h3 className="font-semibold">Number of Initial Data Points:</h3>
                    <p>4 (4 random data points to be sampled from the dataset distribution)</p>
                </div>
                <div>
                    <h3 className="font-semibold">Number of Initial Comparisons:</h3>
                    <p>6 (6 pairwise comparisons to be made between the initial data points)</p>
                </div>
                </div>
            </AccordionContent>
            </AccordionItem>
        </Accordion>
        <Separator orientation="horizontal" className="mt-8" />
        <div className="text-sm text-gray-500 mt-4 flex items-center justify-center">
            <div className="p-2 bg-gray-50 border border-gray-300 rounded-lg">
            <a href="https://github.com/imperorrp" target="_blank" rel="noopener noreferrer">App by Ratish</a>
            </div>
        </div>
    </div>
  );
}
