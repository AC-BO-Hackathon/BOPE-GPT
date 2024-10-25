import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { useBopeStore } from "@/hooks/bopeStore";
import type { BopeState } from "@/hooks/bopeStore";
import { ChevronLeft, ChevronRight } from "lucide-react"

const ITEMS_PER_PAGE = 10;

export function AllDataPointsTable() {
  const { latestBopeData } = useBopeStore();
  const bopeState = latestBopeData?.bope_state as BopeState | null;
  const [currentPage, setCurrentPage] = useState(1);

  if (!bopeState || !bopeState.comparison_data) {
    return <div>No data points available</div>;
  }

  const { comparison_data, input_columns, output_columns } = bopeState;
  const allDataPoints = comparison_data.pair_indices.flat();
  const uniqueDataPoints = Array.from(new Set(allDataPoints));

  const totalPages = Math.ceil(uniqueDataPoints.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentDataPoints = uniqueDataPoints.slice(startIndex, endIndex);

  const goToNextPage = () => setCurrentPage(prev => Math.min(prev + 1, totalPages));
  const goToPrevPage = () => setCurrentPage(prev => Math.max(prev - 1, 1));

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <Table>
          <TableCaption>All Data Points</TableCaption>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[80px]">Data Point ID</TableHead>
              {input_columns.map((col, index) => (
                <TableHead key={`input-${index}`}>{col}</TableHead>
              ))}
              {output_columns.map((col, index) => (
                <TableHead key={`output-${index}`}>{col}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {currentDataPoints.map((dataPointId) => {
              const dataPointIndex = comparison_data.pair_indices.findIndex(pair => pair.includes(dataPointId));
              const isFirstInPair = comparison_data.pair_indices[dataPointIndex][0] === dataPointId;
              const inputValues = comparison_data.pair_input_values[dataPointIndex][isFirstInPair ? 0 : 1];
              const outputValues = comparison_data.pair_output_values[dataPointIndex][isFirstInPair ? 0 : 1];

              return (
                <TableRow key={dataPointId}>
                  <TableCell className="font-medium">{dataPointId}</TableCell>
                  {inputValues.map((value, i) => (
                    <TableCell key={`input-${i}`}>{value.toFixed(4)}</TableCell>
                  ))}
                  {outputValues.map((value, i) => (
                    <TableCell key={`output-${i}`}>{value.toFixed(4)}</TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Page {currentPage} of {totalPages}
        </div>
        <div className="space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={goToPrevPage}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={goToNextPage}
            disabled={currentPage === totalPages}
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
