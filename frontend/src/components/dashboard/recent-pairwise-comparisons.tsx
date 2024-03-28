import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

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

import React from 'react';

//TODO: table column names should be from dataset, not preset to Fischer-Tropsch dataset

//sample table data for past pairwise comparisons
const pairwise_data = [
  {
    id: "01",
    W_FCO: 24.3902439,
    H2_CO: 1,
    Temp: 480,
    pressure: 1,
    X_CO: 0.028510785,
    S_CO2: 0.491949923,
    S_CH4: 0.000193406,
    S_C2C4p: 0.000260784,
    S_C2C4: 0.074070395,
  },
  {
    id: "02",
    W_FCO: 29.26829268,
    H2_CO: 1,
    Temp: 480,
    pressure: 1,
    X_CO: 0.033117823,
    S_CO2: 0.492121555,
    S_CH4: 0.000198438,
    S_C2C4p: 0.000266597,
    S_C2C4: 0.075670925,
  },
  {
    id: "03",
    W_FCO: 34.14634146,
    H2_CO: 1,
    Temp: 480,
    pressure: 1,
    X_CO: 0.037445555,
    S_CO2: 0.492262488,
    S_CH4: 0.000203374,
    S_C2C4p: 0.000272255,
    S_C2C4: 0.077225409,
  },
  {
    id: "04",
    W_FCO: 39.02439024,
    H2_CO: 1,
    Temp: 480,
    pressure: 1,
    X_CO: 0.041528469,
    S_CO2: 0.492386603,
    S_CH4: 0.000208193,
    S_C2C4p: 0.000277741,
    S_C2C4: 0.078729088,
  },
  {
    id: "05",
    W_FCO: 43.90243902,
    H2_CO: 1,
    Temp: 480,
    pressure: 1,
    X_CO: 0.045396013,
    S_CO2: 0.492499905,
    S_CH4: 0.000212887,
    S_C2C4p: 0.00028305,
    S_C2C4: 0.080180665,
  },
  {
    id: "06",
    W_FCO: 48.7804878,
    H2_CO: 1,
    Temp: 480,
    pressure: 1,
    X_CO: 0.049073104,
    S_CO2: 0.492605159,
    S_CH4: 0.000217455,
    S_C2C4p: 0.000288185,
    S_C2C4: 0.081580815,
  },
];


export function RecentPairwiseComparisonsTable() {
    return (
        <div className="space-y-8">
            <Table>
              <TableCaption>List of Recent Pairwise Comparisons</TableCaption>
              <TableHeader>
                <TableRow>
                    <TableHead className="font-medium">Dataset ID</TableHead>
                  <TableHead className="font-medium">W/FCO</TableHead>
                  <TableHead className="font-medium">H2/CO</TableHead>
                  <TableHead className="font-medium">Temp</TableHead>
                  <TableHead className="text-right">Pressure</TableHead>
                  <TableHead className="font-medium">X_CO</TableHead>
                  <TableHead className="font-medium">S_CO2</TableHead>
                  <TableHead className="font-medium">S_CH4</TableHead>
                  <TableHead className="font-medium">S_C2C4p</TableHead>
                  <TableHead className="font-medium">S_C2C4=</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {pairwise_data.map((data, index) => {
                    return (
                    <React.Fragment key={data.id}>
                        <TableRow>
                          <TableCell className="font-medium">{data.id}</TableCell>
                          <TableCell>{data.W_FCO}</TableCell>
                          <TableCell>{data.H2_CO}</TableCell>
                          <TableCell>{data.Temp}</TableCell>
                          <TableCell>{data.pressure}</TableCell>
                          <TableCell>{data.X_CO}</TableCell>
                          <TableCell>{data.S_CO2}</TableCell>
                          <TableCell>{data.S_CH4}</TableCell>
                          <TableCell>{data.S_C2C4p}</TableCell>
                          <TableCell>{data.S_C2C4}</TableCell>
                        </TableRow>
                        {index % 2 === 1 && (
                        <TableRow>
                            <TableCell colSpan={9} className="font-medium">Selected Sample</TableCell>
                            <TableCell className="text-right">ID #x</TableCell>
                        </TableRow>
                        )}
                    </React.Fragment>
                    );
                })}

              </TableBody>
              <TableFooter>
              </TableFooter>
            </Table>
        </div>
    );
}

/*
<div className="flex items-center">
                <Avatar className="h-9 w-9">
                    <AvatarImage src="/avatars/01.png" alt="Avatar" />
                    <AvatarFallback>OM</AvatarFallback>
                </Avatar>
                <div className="ml-4 space-y-1">
                    <p className="text-sm font-medium leading-none">Olivia Martin</p>
                    <p className="text-sm text-muted-foreground">
                        olivia.martin@email.com
                    </p>
                </div>
                <div className="ml-auto font-medium">+$1,999.00</div>
            </div>
            <div className="flex items-center">
                <Avatar className="flex h-9 w-9 items-center justify-center space-y-0 border">
                    <AvatarImage src="/avatars/02.png" alt="Avatar" />
                    <AvatarFallback>JL</AvatarFallback>
                </Avatar>
                <div className="ml-4 space-y-1">
                    <p className="text-sm font-medium leading-none">Jackson Lee</p>
                    <p className="text-sm text-muted-foreground">jackson.lee@email.com</p>
                </div>
                <div className="ml-auto font-medium">+$39.00</div>
            </div>
            <div className="flex items-center">
                <Avatar className="h-9 w-9">
                    <AvatarImage src="/avatars/03.png" alt="Avatar" />
                    <AvatarFallback>IN</AvatarFallback>
                </Avatar>
                <div className="ml-4 space-y-1">
                    <p className="text-sm font-medium leading-none">Isabella Nguyen</p>
                    <p className="text-sm text-muted-foreground">
                        isabella.nguyen@email.com
                    </p>
                </div>
                <div className="ml-auto font-medium">+$299.00</div>
            </div>
            <div className="flex items-center">
                <Avatar className="h-9 w-9">
                    <AvatarImage src="/avatars/04.png" alt="Avatar" />
                    <AvatarFallback>WK</AvatarFallback>
                </Avatar>
                <div className="ml-4 space-y-1">
                    <p className="text-sm font-medium leading-none">William Kim</p>
                    <p className="text-sm text-muted-foreground">will@email.com</p>
                </div>
                <div className="ml-auto font-medium">+$99.00</div>
            </div>
            <div className="flex items-center">
                <Avatar className="h-9 w-9">
                    <AvatarImage src="/avatars/05.png" alt="Avatar" />
                    <AvatarFallback>SD</AvatarFallback>
                </Avatar>
                <div className="ml-4 space-y-1">
                    <p className="text-sm font-medium leading-none">Sofia Davis</p>
                    <p className="text-sm text-muted-foreground">sofia.davis@email.com</p>
                </div>
                <div className="ml-auto font-medium">+$39.00</div>
            </div>
    */