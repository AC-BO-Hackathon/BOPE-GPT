//"use client";
import React, {useRef, useState, useEffect} from 'react';
import dynamic from 'next/dynamic';
import { useBopeStore } from "@/hooks/bopeStore";
import type { ParetoVisualizationData, ParetoPlotData, DataPoint } from "@/hooks/bopeStore";
import {
    CardHeader,
    CardTitle,
    CardContent,
    CardDescription,
    Card,
  } from "@/components/ui/card";

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });


function ParetoPlot({ plot, paretoData, index }: { plot: ParetoPlotData; paretoData: ParetoVisualizationData; index: number }) {
    const [tooltip, setTooltip] = useState({ visible: false, content: '', x: 0, y: 0 });
    const cardRef = useRef<HTMLDivElement>(null);

    const generateTooltipContent = (dataPoint: DataPoint, isParetoOptimal: boolean) => {
        let content = `<div style="font-size: 10px;">`;
        content += `<strong>Point ${dataPoint.id + 1}</strong><br>`;
        content += `Is Pareto: ${isParetoOptimal ? 'Yes' : 'No'}<br><br>`;
        
        content += '<strong>Inputs:</strong><br>';
        for (const [key, value] of Object.entries(dataPoint.input_values)) {
            content += `${key}: ${typeof value === 'number' ? value.toFixed(4) : value}<br>`;
        }
        
        content += '<br><strong>Outputs:</strong><br>';
        for (const [key, value] of Object.entries(dataPoint.output_values)) {
            content += `${key}: ${typeof value === 'number' ? value.toFixed(4) : value}<br>`;
        }
        
        content += '</div>';
        return content;
    };

    const handlePlotHover = (event: Plotly.PlotMouseEvent) => {
        const point = event.points[0];
        if (point && point.pointIndex !== undefined) {
            const dataPoint = paretoData.data_points[point.pointIndex];
            const isParetoOptimal = plot.is_pareto[point.pointIndex] ?? false;
            if (dataPoint) {
                const content = generateTooltipContent(dataPoint, isParetoOptimal);
                setTooltip({
                    visible: true,
                    content,
                    x: point.xaxis?.l2p?.(point.x as number) ?? 0,
                    y: point.yaxis?.l2p?.(point.y as number) ?? 0
                });
            }
        }
    };

    const handlePlotUnhover = () => {
        setTooltip({ ...tooltip, visible: false });
    };

    return (
        <Card className="col-span-1" ref={cardRef}>
            <CardHeader>
                <CardTitle>{plot.x_label} vs {plot.y_label}</CardTitle>
                <CardDescription>
                    
                    Pareto Front {index + 1}
                </CardDescription>
            </CardHeader>
            <CardContent className="relative" style={{ paddingBottom: '100%' }}>
                <div className="absolute inset-0">
                    <Plot
                        data={[
                            {
                                x: plot.point_indices.map(i => paretoData.data_points[i]?.output_values[plot.x_label] ?? null).filter(x => x !== null),
                                y: plot.point_indices.map(i => paretoData.data_points[i]?.output_values[plot.y_label] ?? null).filter(y => y !== null),
                                mode: 'markers',
                                type: 'scatter',
                                marker: {
                                    color: plot.is_pareto.map(isParetoI => isParetoI ? 'red' : 'blue'),
                                    size: 8,
                                },
                                hoverinfo: 'none',
                            },
                        ]}
                        layout={{
                            autosize: true,
                            margin: { l: 50, r: 50, b: 50, t: 50 },
                            xaxis: { 
                                title: plot.x_label,
                                titlefont: { size: 10 },
                            },
                            yaxis: { 
                                title: plot.y_label,
                                titlefont: { size: 10 },
                            },
                            showlegend: false,
                        }}
                        config={{ 
                            responsive: true,
                            displayModeBar: false,
                        }}
                        style={{
                            width: '100%',
                            height: '100%',
                        }}
                        useResizeHandler={true}
                        onHover={handlePlotHover}
                        onUnhover={handlePlotUnhover}
                    />
                    {tooltip.visible && (
                        <div
                            style={{
                                position: 'absolute',
                                left: `${tooltip.x}px`,
                                top: `${tooltip.y}px`,
                                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                                border: '1px solid #ccc',
                                padding: '8px',
                                borderRadius: '4px',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                                zIndex: 1000,
                                pointerEvents: 'none',
                                width: '200px',
                                maxWidth: '250px',
                                overflow: 'auto',
                                maxHeight: '300px',
                            }}
                            dangerouslySetInnerHTML={{ __html: tooltip.content }}
                        />
                    )}
                </div>
            </CardContent>
        </Card>
    );
}

export function ParetoFrontsVisualization() {
    const { latestBopeData } = useBopeStore();
    const paretoData = latestBopeData?.bope_state.pareto_plot_data as ParetoVisualizationData | null;

    if (!paretoData) {
        return <div>No pareto plot data available</div>;
    }

    return (
        <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {paretoData.pareto_plots.map((plot, index) => (
                <ParetoPlot 
                    key={index}
                    plot={plot}
                    paretoData={paretoData}
                    index={index}
                />
            ))}
        </div>
    );
}
