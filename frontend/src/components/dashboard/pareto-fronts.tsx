"use client";

import {
    CardHeader,
    CardTitle,
    CardContent,
    CardDescription,
    Card,
  } from "@/components/ui/card";

export function ParetoFrontsVisualization() {
    return (
        <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => (
                <Card className="col-span-1" key={index}>
                    <CardHeader>
                        <CardTitle>Pareto Front #{index + 1}</CardTitle>
                        <CardDescription>
                            Objective X v/s Objective Y
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div></div>
                    </CardContent>
                </Card>
            ))}
        </div>
    )
}