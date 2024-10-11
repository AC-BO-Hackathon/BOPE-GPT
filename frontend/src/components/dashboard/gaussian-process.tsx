import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useBopeStore } from "@/hooks/bopeStore";
import type { VisualizationDataModel, ContourDataModel } from "@/hooks/bopeStore";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";

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
    //const [plotData, setPlotData] = useState<Plotly.Data[]>([]); del
    const [meanPlotData, setMeanPlotData] = useState<Plotly.Data[]>([]);
    const [uncertaintyPlotData, setUncertaintyPlotData] = useState<Plotly.Data[]>([]);


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
        <div>
            {sliderInputs.map((key) => {
                const data = visualizationData.slider_data[key] as SliderData | undefined;
                if (!data) return null;
                let sliderValue = sliderValues[key] ?? data.default_range[0];
                if (sliderValue === undefined){
                    sliderValue = 0;
                }
                return (
                    <div key={key} className='mb-4 flex flex-row justify-center'>
                        <label className="px-2">Input {parseInt(key.split('_')[1] ?? '0')+1}</label>
                        <Slider
                            min={data.min}
                            max={data.max}
                            step={(data.max - data.min) / (data.default_range.length - 1)}
                            value={[sliderValue]}
                            onValueChange={(value) => handleSliderChange(key, value[0] ?? 0)}
                            className="w-1/2 px-5"
                        />      
                        <span>{sliderValues[key]?.toFixed(4) || (data.default_range[0] ?? 0).toFixed(4)}</span>
                    </div>
                );
            })}

            <div className="flex flex-col lg:justify-between">
                <div className="w-1/2 p-2">
                    <Plot
                        data={meanPlotData}
                        layout={{
                            title: 'Mean Surface',
                            scene: {
                                xaxis: { title: 'Input 0' },
                                yaxis: { title: 'Input 1' },
                                zaxis: { title: 'Mean' },
                            },
                            autosize: true,
                            //height: 100%,
                            //width: '100%',
                        }}
                    />
                </div>

                <div className="w-1/2 p-2">
                    <Plot
                        data={uncertaintyPlotData}
                        layout={{
                            title: 'Uncertainty Heatmap (standard deviation)',
                            xaxis: { title: 'Input 0' },
                            yaxis: { title: 'Input 1' },
                            autosize: true,
                            //height: 600,
                            //width: 800,
                        }}
                    />
                </div>
            </div>
        </div>
    );
}
