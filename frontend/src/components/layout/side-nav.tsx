
"use client";
import Link from "next/link";
import { useBopeStore } from "../../hooks/bopeStore";

import { type NavItem } from "@/types";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useSidebar } from "@/hooks/useSidebar";
import { Button, buttonVariants } from "@/components/ui/button";
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { toast } from "@/components/ui/use-toast"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

import {
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
  Card,
} from "@/components/ui/card";

import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/layout/subnav-accordion";
import { useEffect, useState } from "react";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import { ConstructionIcon } from "lucide-react";
//import { Input } from "postcss";

import type { InitializeBopeResponse, BopeState, IterationBopeResponse } from '../../hooks/bopeStore';

interface SideNavProps {
    items: NavItem[];
    setOpen?: (open: boolean) => void;
    className?: string;
}

interface ErrorData {
  detail: string | { msg: string; type: string }[];
  status_code?: number;
}

const NextIterationFormSchema = z.object({
  llm_prompt: z
    .string().optional(),
    /*.min(10, {
      message: "Prompt must be at least 10 characters.",
    })
    .max(200, {
      message: "Prompt must not be longer than 200 characters.",
    }),*/
  enable_flexible_prompt: z
    .union([z.boolean(), z.number()])
    .transform((val) => Boolean(val))
    .default(false)
    .optional(),
  enable_llm_explanations: z
    .union([z.boolean(), z.number()])
    .transform((val) => Boolean(val))
    .default(false)
    .optional(),
})

const InitializationFormSchema = z.object({
  llm_prompt: z
    .string()
    .min(10, {
      message: "Prompt must be at least 10 characters.",
    })
    .max(200, {
      message: "Prompt must not be longer than 200 characters.",
    }),
  num_inputs: z
    .string()
    .transform((val) => parseInt(val, 10))
    .refine((val) => !isNaN(val), {
      message: "Number of inputs must be a valid number.",
    })
    .pipe(
      z.number().min(1, {
        message: "Number of inputs must be at least 1.",
      }),
    ),
  num_initial_samples: z
    .string()
    .transform((val) => parseInt(val, 10))
    .refine((val) => !isNaN(val), {
      message: "Number of inputs must be a valid number.",
    })
    .pipe(
      z.number().min(4, {
        message: "Number of inputs must be at least 4.",
      }),
    ),
  num_initial_comparisons: z
    .string()
    .transform((val) => parseInt(val, 10))
    .refine((val) => !isNaN(val), {
      message: "Number of inputs must be a valid number.",
    })
    .pipe(
      z.number().min(6, {
        message: "Number of inputs must be at least 6.",
      }),
    ),
  enable_flexible_prompt: z.boolean().default(false).optional(),
  enable_llm_explanations: z.boolean().default(false).optional(),
})


export function SideNav({ items, setOpen, className }: SideNavProps) {
    const { stateId, latestBopeData, loading, uploadedDatasetData, setLoading, setLatestBopeData } = useBopeStore();
    const path = usePathname();
    const { isOpen } = useSidebar();
    const [openItem, setOpenItem] = useState("");
    const [lastOpenItem, setLastOpenItem] = useState("");
    const [userInput, setUserInput] = useState(''); // New state for user input

    const all_columns = uploadedDatasetData?.column_names || []; 

    // if there's no visualization data yet (if the BOPE initialization hasn't been done yet)
    const is_bope_initialized = latestBopeData !== null;
    console.log(latestBopeData);

    useEffect(() => {
        if (isOpen) {
            setOpenItem(lastOpenItem);
        } else {
            setLastOpenItem(openItem);
            setOpenItem("");
        }
    }, [isOpen]);

    const form = useForm<z.infer<typeof InitializationFormSchema>>({
      resolver: zodResolver(InitializationFormSchema),
    })

    const IterationForm = useForm<z.infer<typeof NextIterationFormSchema>>({
      resolver: zodResolver(NextIterationFormSchema),
    })

    async function onNextIterationSubmit(data: z.infer<typeof NextIterationFormSchema>){
      console.log('onNextIterationSubmit called');
      console.log(data);
      setLoading(true);
      toast({
        title: "You submitted the following values:",
        description: (
          <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
            <code className="text-black">{JSON.stringify(data, null, 2)}</code>
          </pre>
        ),
        duration: 5000,
      })
      IterationForm.setValue('llm_prompt', '');
      try {
        const backendUrl = process.env.NEXT_PUBLIC_LOCAL_BACKEND_URL;
        if (!backendUrl) {
          throw new Error('Backend URL is not defined');
        }
        const response = await fetch(`${backendUrl}/run_next_iteration/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ ...data, state_id: stateId }),
        });
  
        if (!response.ok) {
          //throw new Error('Failed to process data');
          const errorData: ErrorData = await response.json() as ErrorData;
          console.error('Error response from server:', errorData);
          throw new Error(`Failed to process data: ${JSON.stringify(errorData)}`);
        }
  
        const result: IterationBopeResponse = await response.json() as IterationBopeResponse;
        if (typeof result.bope_state === 'string') {
          result.bope_state = JSON.parse(result.bope_state) as BopeState;
        }
        setLatestBopeData(result);
        console.log(`IterationBopeResponse: ${JSON.stringify(result, null, 2)}`);
  
        toast({
          title: "Data processed successfully",
          description: "Visualization data has been updated",
          duration: 5000,
        });
      } catch (error) {
        console.error('Error processing data:', error);
        toast({
          title: "Error",
          description: "Failed to process data",
          duration: 5000,
        });
      } finally {
        setLoading(false);
      }
    }
   
    async function onInitializationSubmit(data: z.infer<typeof InitializationFormSchema>){
      console.log(data);
      setLoading(true);
      toast({
        title: "You submitted the following values:",
        description: (
          <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
            <code className="text-black">{JSON.stringify(data, null, 2)}</code>
          </pre>
        ),
        duration: 5000,
      })
      form.setValue('llm_prompt', '');
      form.setValue('num_inputs', 0);
      form.setValue('num_initial_samples', 0);
      form.setValue('num_initial_comparisons', 0);

      try {
        const backendUrl = process.env.NEXT_PUBLIC_LOCAL_BACKEND_URL;
        if (!backendUrl) {
          throw new Error('Backend URL is not defined');
        }
        const response = await fetch(`${backendUrl}/initialize_bope/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ ...data, state_id: stateId }),
        });
  
        if (!response.ok) {
          //throw new Error('Failed to process data');
          const errorData: ErrorData = await response.json() as ErrorData;
          console.error('Error response from server:', errorData);
          throw new Error(`Failed to process data: ${JSON.stringify(errorData)}`);
        }
  
        const result: InitializeBopeResponse = await response.json() as InitializeBopeResponse;
        if (typeof result.bope_state === 'string') {
          result.bope_state = JSON.parse(result.bope_state) as BopeState;
        }
        setLatestBopeData(result);
        console.log(`InitializationBopeResponse: ${JSON.stringify(result, null, 2)}`);
  
        toast({
          title: "Data processed successfully",
          description: "Visualization data has been updated",
          duration: 5000,
        });
      } catch (error) {
        console.error('Error processing data:', error);
        toast({
          title: "Error",
          description: "Failed to process data",
          duration: 5000,
        });
      } finally {
        setLoading(false);
      }

    }


    return (
    <nav className={`relative space-y-2 ${loading ? 'blur-sm' : ''}`}>
      {items.map((item) =>
        item.isChidren ? (
          <Accordion
            type="single"
            collapsible
            className="space-y-2 text-sm"
            key={item.title}
            value={openItem}
            onValueChange={setOpenItem}
          >
            <AccordionItem value={item.title} className="border-none ">
              <AccordionTrigger
                className={cn(
                  buttonVariants({ variant: 'ghost' }),
                  'group relative flex h-12 justify-between px-4 py-2 text-base duration-200 hover:bg-muted hover:no-underline',
                )}
              >
                <div>
                  <item.icon className={cn('h-5 w-5', item.color)} />
                </div>
                <div
                  className={cn(
                    'absolute left-12 text-base duration-200 ',
                    !isOpen && className,
                  )}
                >
                  {item.title}
                </div>

                {isOpen && (
                  <ChevronDownIcon className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200" />
                )}
              </AccordionTrigger>
              <AccordionContent className="mt-2 space-y-4 pb-1">
                {item.children?.map((child) => (
                  <Link
                    key={child.title}
                    href={child.href}
                    onClick={() => {
                      if (setOpen) setOpen(false)
                    }}
                    className={cn(
                      buttonVariants({ variant: 'ghost' }),
                      'group relative flex h-12 justify-start gap-x-3',
                      path === child.href &&
                        'bg-muted font-bold hover:bg-muted',
                    )}
                  >
                    <child.icon className={cn('h-5 w-5', child.color)} />
                    <div
                      className={cn(
                        'absolute left-12 text-base duration-200',
                        !isOpen && className,
                      )}
                    >
                      {child.title}
                    </div>
                  </Link>
                ))}
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : (
          <Link
            key={item.title}
            href={item.href}
            onClick={() => {
              if (setOpen) setOpen(false)
            }}
            className={cn(
              buttonVariants({ variant: 'ghost' }),
              'group relative flex h-12 justify-start',
              path === item.href && 'bg-muted font-bold hover:bg-muted',
            )}
          >
            <item.icon className={cn('h-5 w-5', item.color)} />
            <span
              className={cn(
                'absolute left-12 text-base duration-200',
                !isOpen && className,
              )}
            >
              {item.title}
            </span>
          </Link>
        ),
      )}
      <div>
        { all_columns.length === 0? (
           <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                BOPE-GPT Initialization Panel
              </CardTitle>
            </CardHeader>
            <CardContent className="flex justify-center items-center">
              <div>
                <p className="text-xs py-2 text-red-500 text-muted-foreground">
                  No Dataset detected. Upload one to proceed. 
                </p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Form {...form}>
            {!is_bope_initialized? (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  form
                    .handleSubmit(onInitializationSubmit)()
                    .catch((err) => {
                      // Handle the error
                      console.error(err);
                    });
                }}
                className="space-y-8"
                >
                <FormField
                  control={form.control}
                  name="llm_prompt"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>BOPE-GPT Initialization Panel</FormLabel>
                      <div className="w-full">
                        <p className="text-xs text-center text-green-500 w-full">Dataset processed</p>
                      </div>
                      <FormControl>
                        <Textarea
                          placeholder="Enter LLM Prompt"
                          className="resize-none h-80x"
                          {...field}
                        />
                      </FormControl>
                      <FormDescription className="px-2 text-xs">
                        (Used by the Cohere LLM model to make pairwise comparisons)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="num_inputs"
                  render={({ field }) => (
                    <FormItem>
                      {
                      //<FormLabel>Enter initialization variables</FormLabel>
                      }
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="Number of input features"
                          className="resize-none h-10"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="num_initial_samples"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="Number of initial samples"
                          className="resize-none h-10"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="num_initial_comparisons"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="Number of initial pairwise comparisons"
                          className="resize-none h-10"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                    control={form.control}
                    name="enable_flexible_prompt"
                    render={({ field }) => (
                      <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                        <div className="space-y-0.5">
                          <FormLabel className="text-base">
                            Flexible Prompt Setup
                          </FormLabel>
                          <FormDescription>
                            Enable flexible prompt setup across iterations
                          </FormDescription>
                        </div>
                        <FormControl>
                          <Switch
                            checked={field.value}
                            onCheckedChange={field.onChange}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="enable_llm_explanations"
                    render={({ field }) => (
                      <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                        <div className="space-y-0.5">
                          <FormLabel className="text-base">
                            LLM Explanations
                          </FormLabel>
                          <FormDescription>
                            Enable pairwise comparison explanations from LLM
                          </FormDescription>
                        </div>
                        <FormControl>
                          <Switch
                            checked={field.value}
                            onCheckedChange={field.onChange}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                <div className="flex justify-center">
                  <Button type="submit">Initialize BOPE Modelling</Button>
                </div>
                </form>
              ) : (
                <form
                onSubmit={(e) => {
                  console.log('next iteration button clicked');
                  e.preventDefault();
                  console.log('IterationForm state:', IterationForm.getValues());
                  IterationForm
                    .handleSubmit(
                      //onNextIterationSubmit
                    //)()
                    (data) => {
                      console.log('Form is valid, calling onNextIterationSubmit with data:', data);
                      return onNextIterationSubmit(data);
                    },
                    (errors) => {
                      console.error('Form validation failed:', errors);
                    })()
                    .then(() => {
                      console.log('Form submission completed successfully');
                    })
                    .catch((err) => {
                      // Handle the error
                      console.error(err);
                    });
                }}
                className="space-y-8"
                >
                <FormField
                  control={IterationForm.control}
                  name="llm_prompt"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Run Next BOPE-GPT Model Iteration</FormLabel>
                      <FormControl>
                        <Textarea
                          placeholder="Enter LLM Prompt" // This should not appear by default, only if 'flexible LLM prompts' is true
                          className="resize-none h-80x"
                          {...field}
                        />
                      </FormControl>
                      <FormDescription className="px-2 text-xs">
                        (Used by the Cohere LLM model to make pairwise comparisons)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                    control={IterationForm.control}
                    name="enable_flexible_prompt"
                    render={({ field }) => (
                      <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                        <div className="space-y-0.5">
                          <FormLabel className="text-base">
                            Flexible Prompt Setup
                          </FormLabel>
                          <FormDescription>
                            Enable flexible prompt setup across iterations
                          </FormDescription>
                        </div>
                        <FormControl>
                          <Switch
                            checked={field.value}
                            //onCheckedChange={field.onChange}
                            onCheckedChange={(checked: boolean) => {
                              field.onChange(checked);
                              console.log('LLM explanations changed:', checked);
                            }}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={IterationForm.control}
                    name="enable_llm_explanations"
                    render={({ field }) => (
                      <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                        <div className="space-y-0.5">
                          <FormLabel className="text-base">
                            LLM Explanations
                          </FormLabel>
                          <FormDescription>
                            Enable pairwise comparison explanations from LLM
                          </FormDescription>
                        </div>
                        <FormControl>
                          <Switch
                            checked={field.value}
                            //onCheckedChange={field.onChange}
                            onCheckedChange={(checked: boolean) => {
                              field.onChange(checked);
                              console.log('LLM explanations changed:', checked);
                            }}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                <div className="flex justify-center">
                  <Button type="submit">Next Iteration</Button>
                </div>
            </form>
          )
          }</Form>
        )
      }
      </div>
    </nav>
    );
}
