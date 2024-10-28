import React from "react";
import { useBopeStore } from "../hooks/bopeStore";

import {
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
  Card,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { GaussianProcessVisualization } from "@/components/dashboard/gaussian-process";
import { ParetoFrontsVisualization } from "@/components/dashboard/pareto-fronts";
import { PairwiseComparisonsTable } from "@/components/dashboard/pairwise-comparisons";
import { AllDataPointsTable } from "@/components/dashboard/all-data-points";
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/use-toast"

import type { UploadDatasetSuccessResponse } from "../hooks/bopeStore";

const Home = () => {
  const { latestBopeData, uploadedDatasetData, loading, setStateId, setUploadedDatasetData } = useBopeStore();

  const iterationNumber = latestBopeData?.bope_state.iteration || 0;
  const data_points = latestBopeData?.bope_state.X.length || 0;
  const totalComparisonsMade = latestBopeData?.bope_state.comparisons.length || 0;
  const bestVals = latestBopeData?.bope_state.best_val || [];
  const input_columns = latestBopeData?.bope_state.input_columns || [];
  const iteration_duration = latestBopeData?.bope_state.last_iteration_duration || null;
  // if there's no visualization data yet (if the BOPE initialization hasn't been done yet)
  const is_bope_initialized = latestBopeData !== null;
  const all_columns = uploadedDatasetData?.column_names || []; 

  const { toast } = useToast();  

  const testToast = () => {
    console.log("Attempting to show toast");
    toast({
      title: "Test Toast",
      description: "This is a test toast message.",
    });
    console.log("Toast function called");
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    console.log("Sending uploaded file to backend...");

    try {
      const backendUrl = process.env.NEXT_PUBLIC_LOCAL_BACKEND_URL;
      if (!backendUrl) {
        throw new Error('Backend URL is not defined');
      }
      const response = await fetch(`${backendUrl}/upload_dataset/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload file');
      }

      const result = await response.json() as UploadDatasetSuccessResponse;

      console.log('File uploaded successfully:', result);

      toast({
        title: "Form Submitted",
        description: "Your form has been successfully submitted!",
      });

      // Save state_id to Zustand store
      setStateId(result.state_id);

      // Save `upload_dataset` api call response to Zustand store 
      setUploadedDatasetData(result);

      console.log(`State_id saved: ${result.state_id}`)
    } catch (error) {
      console.error('Error uploading file:', error);

      toast({
        title: "Upload Failed",
        description: "There was an error uploading your file. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleFileUpload(event).catch(error => {
      console.error('Error in handleFileUpload:', error);
    });
  };


  return (
    <div>
    {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-75 z-50">
            <div className="loader">Loading...</div>
          </div>
        )}
    <div className={`flex h-full flex-col relative ${loading ? 'blur-sm' : ''}`}>
      <div className="flex-1 space-y-4 p-8 pt-6">
        <div className="flex items-center justify-between space-y-2">
          <h2 className="text-2xl font-bold tracking-tight">Dashboard</h2>
          <div className="flex items-center space-x-2">
            <Label className="px-2 font-bold" htmlFor="uploadDataset">
            {all_columns.length > 0? (
              "Dataset Uploaded"
            ) : ( 
              "Upload Dataset"
            )}
            </Label>
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <Input id="upload_dataset" type="file" onChange={handleChange} />
            </div>
          </div>
        </div>
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="about">
              About BOPE-GPT
            </TabsTrigger>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="visualization">
              Visualizations
            </TabsTrigger>
            <TabsTrigger value="data_points">
              Current Model Data
            </TabsTrigger>
          </TabsList>
          <TabsContent value="about">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>Welcome to BOPE-GPT</CardTitle>
                  <CardDescription>
                    <br></br>
                    An Interface for the BOPE process with Automated Pairwise Comparisons through an LLM. 
                    <br></br>
                    <br></br>
                    (Read more about the origin of BOPE-GPT at: <a href="https://github.com/AC-BO-Hackathon/BOPE-GPT" target="_blank" rel="noopener noreferrer" className="text-blue-500">https://github.com/AC-BO-Hackathon/BOPE-GPT</a>)
                    <br></br>
                    <br></br>
                    <br></br>
                    <p className="font-bold">What is the BOPE process?</p>
                    <br></br>
                    BOPE, or <a href="https://botorch.org/tutorials/bope" target="_blank" rel="noopener noreferrer" className="text-blue-500">Bayesian Optimization via Preference Exploration</a>, is a machine learning technique 
                    for finding potential optimums of a distribution of data that uses pairwise comparisons of data points sampled from this distribution to guide this optimization process. 
                    <br></br>
                    <br></br>
                    This is useful for situations where specific priorities of outputs- the optimization goals- cannot be defined well numerically, or changes,
                    which is often the case in some complicated multi-objective problems. In these scenarios, a human (or human-like) evaluator can express a preference between two outputs and this preference can be used to guide the 
                    optimization process instead.
                    <br></br>
                    <br></br>
                    This &quot;preference&quot; can also be picked through an LLM (the human-like evaluator) if correctly prompted- which is what BOPE-GPT does. 
                    <br></br>
                    <br></br>
                    After each iteration, the list of explored data points expands- Pareto plot and preference visualizations of the pairwiseGP model representing these can then be used 
                    to locate optimal points in the distribution by a user. 
                    <br></br>
                    <br></br>
                    <br></br>
                    <p className="font-bold">Who&apos;s this for?</p>
                    <br></br>
                    This is for people who want to optimize for a multi-objective system they&apos;re trying to model and already have a distribution of input-output data 
                    in a dataset, but don&apos;t have a good way to define the optimization goals numerically- although they can in natural language. 
                    <br></br>
                    <br></br>
                    The BOPE-GPT process is a useful interface for this kind of optimization, and has built in visualizations to help see the latent utility defined in natural language 
                    and select potential optimums.  
                    <br></br>
                    <br></br>
                    A chemical process with tradeoffs in outputs and for which data has already been tabulated, like the Fischer-Tropsch synthesis process, is one potential use case. 
                    <br></br>
                    <br></br>
                    <p className="font-bold">How does BOPE-GPT work?</p>
                    <br></br>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>BOPE-GPT first awaits an upload of an initial CSV dataset- tabular, with columns representing features and rows representing individual data points</li>
                      <li>To initialize the BOPE Process, the Initialization Panel must be filled out: an LLM prompt, the number of dataset columns to use as inputs (outputs defined by elimination), the number of initial data points to sample from the dataset distribution and the number of initial pairwise comparisons to make</li>
                      <li>A neural network to internally represent the distribution of the uploaded dataset is created, mapping inputs to outputs. This is necessary to interpolate between dataset values when necessary</li>
                      <li>The BOPE process begins, iteratively sampling new data points from the distribution and comparing them. A PairwiseGP model is used to model data preference.</li>
                      <li>An LLM is used to automate the pairwise comparisons between data points</li>
                      <li>Visualizations of the current state of the BOPE process are updated as it runs- the PairwiseGP model and the Pareto Fronts of the data points explored so far.</li>
                      <li>Overviews, visualizations and data and comparisons made so far can be viewed via the other tabs on this dashboard after each iteration</li>
                      <li>Each iteration improves the PairwiseGP model and likelihood of finding more optimal data points</li>
                    </ul>
                    <br></br>
                    <br></br>
                    <p className="font-bold">Example BOPE-GPT Initialization Panel Values:</p>
                    <br></br>
                    <div className="pl-4">
                    <p className="font-semibold">LLM Prompt:</p>
                    To maximize all outputs, something like this could be used: <p className="">
                      &quot;Suppose you&apos;re managing a Fischer-Tropsch synthesis process. The four outputs are equally important, and we want to maximize all of them&quot;
                    </p>
                    <br></br>
                    <p className="font-semibold">Number of Input Features:</p>
                    <p className="">
                      4 (the first four columns of the uploaded dataset)
                    </p>
                    <br></br>
                    <p className="font-semibold">Number of Initial Data Points:</p>
                    <p className="">
                      4 (4 random data points to be sampled from the dataset distribution- through the neural network representing it)
                    </p>
                    <br></br>
                    <p className="font-semibold">Number of Initial Comparisons:</p>
                    <p className="">
                      6 (6 pairwise comparisons to be made between the initial data points to gauge relative preference)
                    </p>

                    </div>
                    <br></br>
                    <br></br>
                    <br></br>
                  </CardDescription>
                </CardHeader>
                <CardContent className="pl-2 flex flex-col">
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="visualization">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>Gaussian Process</CardTitle>
                  <CardDescription>
                    PairwiseGP Model Representation
                  </CardDescription>
                </CardHeader>
                <CardContent className="pl-2">
                  <GaussianProcessVisualization />
                </CardContent>
              </Card>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7 py-4">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>Pareto Fronts</CardTitle>
                  <CardDescription>
                    Data Points represented as Pareto fronts
                  </CardDescription>
                </CardHeader>
                <CardContent className="pl-2">
                <ParetoFrontsVisualization />
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="data_points" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7 overflow-x-auto">
                <CardHeader>
                  <CardTitle>Explored Data Points</CardTitle>
                  <CardDescription>
                    All data points sampled so far by the model
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <AllDataPointsTable />  
                </CardContent>
              </Card>
              <Card className="col-span-7 overflow-x-auto">
                <CardHeader>
                  <CardTitle>Pairwise Comparisons</CardTitle>
                  <CardDescription>
                    Data point preferences picked by LLM based on entered prompt
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <PairwiseComparisonsTable />
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="overview" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {/*is_bope_initialized? (
                      "Input Attributes"
                    ) : (
                      "All Dataset Attributes"
                    )*/}
                    Dataset Attributes
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center items-center">
                  {is_bope_initialized? (
                    <div className="text-xs font-bold">
                      <ul className="list-disc py-2 pl-5">
                        {all_columns.map((val, index) => (
                          <li className="py-1" key={index}>{val}{" "}
                            <span className="font-normal text-muted-foreground">
                              {input_columns.includes(val) ? "(input)" : "(output)"}
                            </span>
                          </li>
                        ))}
                      </ul> 
                    </div>
                  ) : (
                    all_columns.length > 0? (
                      <div className="text-xs font-bold">
                        <ul className="list-disc py-2 pl-5">
                          {all_columns.map((val, index) => (
                            <li className="py-1" key={index}>{val}
                            </li>
                          ))}
                        </ul>
                        <div className="text-xs font-normal py-2 text-muted-foreground">
                          Subselect 1 to X of these input features from these by entering X in the initialization panel
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-2xl font-bold">N/A</div>
                        <div className="text-xs py-2 text-muted-foreground">
                          No Dataset Detected
                        </div>
                      </div>
                    )
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Iteration
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center items-center">
                  {is_bope_initialized? (
                    <div>
                    <div className="text-2xl font-bold">{iterationNumber}</div>
                    <p className="text-xs py-2 text-muted-foreground">
                      Took {iteration_duration} seconds
                    </p>
                    </div>
                  ) : (
                    <div>
                        <div className="text-2xl font-bold">0</div>
                        <div className="text-xs py-2 text-muted-foreground">
                          No Model Initialized
                        </div>
                      </div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Data Points in Model 
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center items-center">
                {is_bope_initialized? (
                  <div>
                    <div className="text-2xl font-bold">{data_points}</div>
                    <p className="text-xs py-2 text-muted-foreground">
                      Total sampled data points for the PairwiseGP Model
                    </p>
                  </div>
                  ) : (
                    <div>
                      <div className="text-xl font-bold">N/A</div>
                      <p className="text-xs py-2 text-muted-foreground">
                        No Model Initialized
                      </p>
                  </div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Total Comparisons Made
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center items-center">
                  {is_bope_initialized? (
                    <div>
                      <div className="text-2xl font-bold">{totalComparisonsMade}</div>
                      <p className="text-xs py-2 text-muted-foreground">
                        Pairwise comparisons/relative rankings of sampled data for the model
                      </p>
                    </div>
                  ) : (
                    <div>
                      <div className="text-xl font-bold">N/A</div>
                      <p className="text-xs py-2 text-muted-foreground">
                        No Model Initialized
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Best Output Values</CardTitle>
                </CardHeader>
                <CardContent>
                  {bestVals.length > 0 ? (
                    <div className="text-xs font-bold">
                      <ul className="list-disc">
                        {bestVals.map((val, index) => (
                          <li className="py-1" key={index}>{val}
                            <span className="text-xs font-normal text-muted-foreground">
                            {` (${input_columns[index] || ''})`}
                          </span>
                          </li>
                        ))}
                      </ul> 
                    </div>
                    ) : (
                      <div>
                        <div className="text-xl font-bold">N/A</div>
                        <div className="text-xs py-2 text-muted-foreground py-2">
                          No model initialized
                        </div>
                      </div>
                    )
                  }
                  {/*<p className="text-xs text-muted-foreground">
                    +19% 
                  </p>*/}
                </CardContent>
              </Card>
            </div>
            {/*
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>Latest Pairwise Comparisons</CardTitle>
                  <CardDescription>
                    Picked by LLM last iteration
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RecentPairwiseComparisonsTable />
                </CardContent>
              </Card>
            </div>
            */
            }
          </TabsContent>
        </Tabs>
      </div>
    </div>
    </div>
  );
};

export default Home;
