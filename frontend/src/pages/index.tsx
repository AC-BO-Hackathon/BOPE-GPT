import React from "react";
import { useBopeStore } from "../hooks/bopeStore";

import {
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
  Card,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { CalendarDateRangePicker } from "@/components/dashboard/date-range-picker";
import { GaussianProcessVisualization } from "@/components/dashboard/gaussian-process";
import { AcquisitionFunctionVisualization } from "@/components/dashboard/acquisition-function";
import { ParetoFrontsVisualization } from "@/components/dashboard/pareto-fronts";
import { RecentPairwiseComparisonsTable } from "@/components/dashboard/recent-pairwise-comparisons";
import { PairwiseComparisonsTable } from "@/components/dashboard/pairwise-comparisons";
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/use-toast"

//getting ui components for table
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { string } from "zod";
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
            <Label className="px-2" htmlFor="uploadDataset">Upload Dataset</Label>
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <Input id="upload_dataset" type="file" onChange={handleChange} />
            </div>
          </div>
        </div>
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="visualization">
              Visualizations
            </TabsTrigger>
            <TabsTrigger value="comparisons">
              Pairwise Comparisons
            </TabsTrigger>
            <TabsTrigger value="past" disabled>
              Past Iteration Metrics
            </TabsTrigger>
            <TabsTrigger value="about" disabled>
              About BOPE-GPT
            </TabsTrigger>
          </TabsList>
          <TabsContent value="visualization">
            <ParetoFrontsVisualization />
          </TabsContent>
          <TabsContent value="comparisons" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>All Pairwise Comparisons</CardTitle>
                  <CardDescription>
                    Picked by LLM so far
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
                          Subselect input features from these in the initialization menu
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
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>Gaussian Process</CardTitle>
                </CardHeader>
                <CardContent className="pl-2">
                  <GaussianProcessVisualization />
                </CardContent>
              </Card>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-7">
                <CardHeader>
                  <CardTitle>Acquisition Function</CardTitle>
                </CardHeader>
                <CardContent className="pl-2">
                  <AcquisitionFunctionVisualization />
                </CardContent>
              </Card>
            </div>
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
          </TabsContent>
        </Tabs>
      </div>
    </div>
    </div>
  );
};

export default Home;
