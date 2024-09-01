import React from "react";

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

const Home = () => {
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    console.log("Sending uploaded file to backend...");

    try {
      const response = await fetch('http://127.0.0.1:8000/upload_dataset/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload file');
      }

      const result = await response.json();
      console.log('File uploaded successfully:', result);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleFileUpload(event).catch(error => {
      console.error('Error in handleFileUpload:', error);
    });
  };

  return (
    <div className="flex h-full flex-col">
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
            <TabsTrigger value="analysis">
              More Analysis
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
          <TabsContent value="analysis">
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
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Iteration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">32</div>
                  <p className="text-xs text-muted-foreground">
                    Completed in XYZ seconds
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Total Comparisons Made
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">64</div>
                  <p className="text-xs text-muted-foreground">
                    Abcd...
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Progress Placeholder</CardTitle>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    className="h-4 w-4 text-muted-foreground"
                  >
                    <rect width="20" height="14" x="2" y="5" rx="2" />
                    <path d="M2 10h20" />
                  </svg>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">+12,234</div>
                  <p className="text-xs text-muted-foreground">
                    +19% 
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Progress Placeholder
                  </CardTitle>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    className="h-4 w-4 text-muted-foreground"
                  >
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                  </svg>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">+573</div>
                  <p className="text-xs text-muted-foreground">
                    +201 since last hour
                  </p>
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
  );
};

export default Home;
