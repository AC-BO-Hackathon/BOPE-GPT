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
import type { ComparisonDataModel, BopeState } from "@/hooks/bopeStore";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react"

const ITEMS_PER_PAGE = 10;

export function PairwiseComparisonsTable() {
  const { latestBopeData } = useBopeStore();
  const bopeState = latestBopeData?.bope_state as BopeState | null;
  const [currentPage, setCurrentPage] = useState(1);

  if (!bopeState || !bopeState.comparison_data) {
    return <div>No comparison data available</div>;
  }

  const { comparison_data, input_columns, output_columns } = bopeState;

  const totalPages = Math.ceil(comparison_data.pair_indices.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentComparisons = comparison_data.pair_indices.slice(startIndex, endIndex);

  const goToNextPage = () => setCurrentPage(prev => Math.min(prev + 1, totalPages));
  const goToPrevPage = () => setCurrentPage(prev => Math.max(prev - 1, 1));

  
  return (
    <div className="space-y-4">
      <Table>
        <TableCaption>List of all pairwise comparisons picked by LLM so far</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">Comparison #</TableHead>
            <TableHead className="w-[100px]">Data Point ID</TableHead>
            {input_columns.map((col, index) => (
              <TableHead key={`input-${index}`}>{col}</TableHead>
            ))}
            {output_columns.map((col, index) => (
              <TableHead key={`output-${index}`}>{col}</TableHead>
            ))}
            <TableHead className="w-[100px] text-right">Preferred</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {currentComparisons.map((pair, index) => (
            <React.Fragment key={startIndex + index}>
              <TableRow className="border-t-2 border-primary/20">
                <TableCell rowSpan={2} className="font-medium text-center align-middle">
                  {startIndex + index + 1}
                </TableCell>
                <TableCell className="font-medium">{pair[0]}</TableCell>
                {comparison_data?.pair_input_values?.[startIndex + index]?.[0]?.map((value, i) => (
                  <TableCell key={`input-a-${i}`}>{value.toFixed(4)}</TableCell>
                ))}
                {comparison_data?.pair_output_values?.[startIndex + index]?.[0]?.map((value, i) => (
                  <TableCell key={`output-a-${i}`}>{value.toFixed(4)}</TableCell>
                ))}
                <TableCell rowSpan={2} className="text-right align-middle">
                  <span className={cn(
                    "px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800",
                    pair[0]
                  )}>
                    {pair[0]}
                  </span>
                </TableCell>
              </TableRow>
              <TableRow className="border-b-2 border-primary/20">
                <TableCell className="font-medium">{pair[1]}</TableCell>
                {comparison_data?.pair_input_values?.[startIndex + index]?.[1]?.map((value, i) => (
                  <TableCell key={`input-b-${i}`}>{value.toFixed(4)}</TableCell>
                ))}
                {comparison_data?.pair_output_values?.[startIndex + index]?.[1]?.map((value, i) => (
                  <TableCell key={`output-b-${i}`}>{value.toFixed(4)}</TableCell>
                ))}
              </TableRow>
            </React.Fragment>
          ))}
        </TableBody>
      </Table>
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

