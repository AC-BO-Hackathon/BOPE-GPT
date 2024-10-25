import React, { useRef, useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useBopeStore } from "@/hooks/bopeStore";
import type { VisualizationDataModel, ContourDataModel } from "@/hooks/bopeStore";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";
import {
    CardHeader,
    CardTitle,
    CardContent,
    CardDescription,
    Card,
  } from "@/components/ui/card";

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SliderData {
    min: number;
    max: number;
    default_range: number[];
}

    
export function GaussianProcessVisualization(): JSX.Element {
    const { latestBopeData } = useBopeStore();
    const visualizationData = latestBopeData?.bope_state.visualization_data as VisualizationDataModel | null;

    const [sliderValues, setSliderValues] = useState<{ [key: string]: number }>({});
    const [meanPlotData, setMeanPlotData] = useState<Plotly.Data[]>([]);
    const [uncertaintyPlotData, setUncertaintyPlotData] = useState<Plotly.Data[]>([]);

    const [containerWidth, setContainerWidth] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const updateWidth = () => {
            if (containerRef.current) {
                setContainerWidth(containerRef.current.offsetWidth);
            }
        };

        updateWidth();
        window.addEventListener('resize', updateWidth);
        return () => window.removeEventListener('resize', updateWidth);
    }, []);

    const plotHeight = Math.max(400, Math.min(600, containerWidth * 0.6));

    // Function to find the closest contour key based on slider values
    const findClosestContourKey = (updatedSliderValues: { [key: string]: number }) => {
        if (!visualizationData) return;

        const contourKeys = Object.keys(visualizationData.contour_data);

        // Find the closest matching contour key
        const closestKey = contourKeys.reduce((closest, key) => {
            // Extract the tensor values from the key (in form input_0_1_tensor([0.0000, 0.1111]))
            const tensorMatch = key.match(/tensor\(\[(.*?)\]\)/);  // Regex to extract values inside tensor
            if (!tensorMatch || !tensorMatch[1]) return closest;

            // Split the values, remove any excess whitespace, and parse them as numbers
            const keyValues = tensorMatch[1]
                .split(',')
                .map(val => parseFloat(val.trim()));

            // Calculate the difference between slider values and key values
            const diff = keyValues.reduce((sum: number, keyValue: number, idx: number) => {
                const sliderKey = `input_${idx + 2}`;  // Use slider values for input_2, input_3
                const sliderValue = updatedSliderValues[sliderKey] ?? 0; // Provide a default value of 0 if undefined
                return sum + Math.abs(keyValue - sliderValue);
            }, 0);

            // If this key has a smaller diff, update closest
            if (diff < closest.diff) {
                return { key, diff };
            }
            return closest;
        }, { key: '', diff: Infinity });

        return closestKey.key;
    };


    const generatePlotData = (updatedSliderValues: { [key: string]: number }) => {
        if (!visualizationData) return;

        const closestContourKey = findClosestContourKey(updatedSliderValues);
        if (!closestContourKey) return;
        console.log(`Closest matching key: ${closestContourKey}`);

        const updatedContourData = visualizationData.contour_data[closestContourKey];
        if (!updatedContourData) return;

        // Update both mean and uncertainty data
        setMeanPlotData([
            {
                x: updatedContourData.x,
                y: updatedContourData.y,
                z: updatedContourData.mean[0],
                type: 'surface',
                colorscale: 'Viridis',
                opacity: 0.9,
                name: 'Mean Surface',
            } as Plotly.Data,
        ]);

        setUncertaintyPlotData([
            {
                x: updatedContourData.x.map((arr: number[]) => arr[0]),
                y: updatedContourData.y[0],
                z: updatedContourData.std[0],
                type: 'heatmap',
                colorscale: 'Bluered',
                opacity: 0.6,
                name: 'Uncertainty Heatmap',
                showscale: true,
            } as Plotly.Data,
        ]);
    };

    useEffect(() => {
        if (visualizationData) {
            // Initialize slider values for non-primary inputs 
            const sliderInputs = Object.keys(visualizationData.slider_data).slice(2); 
            const initialSliderValues = sliderInputs.reduce<{ [key: string]: number }>((acc, key) => {
                const sliderData = visualizationData.slider_data[key] as SliderData;
                acc[key] = sliderData?.default_range[0] ?? 0; // Set to first value of default_range
                return acc;
            }, {});
            setSliderValues(initialSliderValues);

            // Set initial plot data based on initial slider values
            generatePlotData(initialSliderValues);
        }
    }, [visualizationData]);

    const handleSliderChange = (inputKey: string, newValue: number) => {
        const updatedSliderValues = { ...sliderValues, [inputKey]: newValue };
        setSliderValues(updatedSliderValues);

        // Update plot data when slider value changes
        generatePlotData(updatedSliderValues);
    };

    if (!visualizationData) {
        return <div>N/A (No model initialized)</div>;
    }

    const sliderInputs = Object.keys(visualizationData.slider_data).slice(2); // Inputs from 2 onward
    //console.log(`sliderInputs: ${sliderInputs}`);


    return (
        <div className="w-full" ref={containerRef}>
            <Card className="w-full">
                <CardHeader>
                    <CardTitle>Gaussian Process Visualization</CardTitle>
                    <CardDescription>Adjust sliders to see changes in the plots</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col space-y-6">
                    <div className="space-y-4 w-full">
                        {sliderInputs.map((key) => {
                            const data = visualizationData.slider_data[key] as SliderData | undefined;
                            if (!data) return null;
                            let sliderValue = sliderValues[key] ?? data.default_range[0];
                            if (sliderValue === undefined){
                                sliderValue = 0;
                            }
                            return (
                                <div key={key} className='mb-4 flex flex-row justify-between items-center w-full'>
                                    <label className="w-24 flex-shrink-0">Input {parseInt(key.split('_')[1] ?? '0')+1}</label>
                                    <Slider
                                        min={data.min}
                                        max={data.max}
                                        step={(data.max - data.min) / (data.default_range.length - 1)}
                                        value={[sliderValue]}
                                        onValueChange={(value) => handleSliderChange(key, value[0] ?? 0)}
                                        className="flex-grow mx-4"
                                    />      
                                    <span className="w-20 text-right">{sliderValues[key]?.toFixed(4) || (data.default_range[0] ?? 0).toFixed(4)}</span>
                                </div>
                            );
                        })}
                    </div>

                    <div className="w-full" style={{ height: `${plotHeight}px` }}>
                        <Plot
                            data={meanPlotData}
                            layout={{
                                title: 'Mean Surface',
                                autosize: true,
                                scene: {
                                    xaxis: { title: 'Input 1' },
                                    yaxis: { title: 'Input 2' },
                                    zaxis: { title: 'Mean' },
                                },
                                margin: { l: 0, r: 0, b: 0, t: 40 },
                                annotations: [
                                    {
                                        x: 0.5,
                                        y: -0.2,
                                        xref: 'paper',
                                        yref: 'paper',
                                        showarrow: false,
                                        text: 'Mean values here represent utility scores. \n(Higher utility = more relative preference and vice-versa)',
                                        font: {
                                            size: 10,
                                            color: 'black'
                                        },
                                        opacity: 0.8,
                                    },
                                ],
                            }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler={true}
                            config={{ responsive: true }}
                        />
                    </div>

                    <div className="w-full" style={{ height: `${plotHeight}px` }}>
                        <Plot
                            data={uncertaintyPlotData}
                            layout={{
                                title: 'Uncertainty Heatmap',
                                autosize: true,
                                xaxis: { title: 'Input 1' },
                                yaxis: { title: 'Input 2' },
                                margin: { l: 50, r: 50, b: 50, t: 40 },
                                annotations: [
                                    {
                                        x: 0.5,
                                        y: -0.3,
                                        xref: 'paper',
                                        yref: 'paper',
                                        showarrow: false,
                                        text: 'Uncertainty values represent confidence in predictions. (Lower std = more confidence and vice-versa)',
                                        font: {
                                            size: 10,
                                            color: 'black'
                                        },
                                        opacity: 0.8,
                                    },
                                ],
                            }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler={true}
                            config={{ responsive: true }}
                        />
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
