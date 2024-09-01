
"use client";
import Link from "next/link";

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
import { toast } from "@/components/ui/use-toast"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"


import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/layout/subnav-accordion";
import { useEffect, useState } from "react";
import { ChevronDownIcon } from "@radix-ui/react-icons";

interface SideNavProps {
    items: NavItem[];
    setOpen?: (open: boolean) => void;
    className?: string;
}

const FormSchema = z.object({
  prompt: z
    .string()
    .min(10, {
      message: "Prompt must be at least 10 characters.",
    })
    .max(200, {
      message: "Prompt must not be longer than 200 characters.",
    }),
  num_inputs: z
    .string()
    .min(1, {
      message: "Number must be at least 1 characters.",
    })
    .max(10, {
      message: "Number must not be longer than 10 characters.",
    }),
  num_init_samples: z
    .string()
    .min(1, {
      message: "Number must be at least 1 characters.",
    })
    .max(10, {
      message: "Number must not be longer than 10 characters.",
    }),
  size_batch: z
    .string()
    .min(1, {
      message: "Number must be at least 1 characters.",
    })
    .max(10, {
      message: "Number must not be longer than 10 characters.",
    }),
  enable_flexible_prompt: z.boolean().default(false).optional(),
  enable_llm_explanations: z.boolean().default(false).optional(),
})


export function SideNav({ items, setOpen, className }: SideNavProps) {
    const path = usePathname();
    const { isOpen } = useSidebar();
    const [openItem, setOpenItem] = useState("");
    const [lastOpenItem, setLastOpenItem] = useState("");
    const [userInput, setUserInput] = useState(''); // New state for user input

    useEffect(() => {
        if (isOpen) {
            setOpenItem(lastOpenItem);
        } else {
            setLastOpenItem(openItem);
            setOpenItem("");
        }
    }, [isOpen]);

    const form = useForm<z.infer<typeof FormSchema>>({
      resolver: zodResolver(FormSchema),
    })
   
    function onSubmit(data: z.infer<typeof FormSchema>){
      console.log(data);
      toast({
        title: "You submitted the following values:",
        description: (
          <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
            <code className="text-black">{JSON.stringify(data, null, 2)}</code>
          </pre>
        ),
        duration: 5000,
      })
      form.setValue('prompt', '');
      form.setValue('num_inputs', '');
      form.setValue('num_init_samples', '');
      form.setValue('size_batch', '');
    }


    return (
        <nav className="space-y-2">
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
      <Form {...form}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            form
              .handleSubmit(onSubmit)()
              .catch((err) => {
                // Handle the error
                console.error(err);
              });
          }}
          className="space-y-8"
          >
          <FormField
            control={form.control}
            name="prompt"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Initialize BOPE-GPT Parameters</FormLabel>
                <FormControl>
                  <Textarea
                    placeholder="Enter LLM Prompt"
                    className="resize-none h-80x"
                    {...field}
                  />
                </FormControl>
                <FormDescription>
                  Initial prompt setup for the LLM
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
                <FormLabel>Enter input variables</FormLabel>
                <FormControl>
                  <Textarea
                    placeholder="Number of inputs"
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
            name="num_init_samples"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Textarea
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
            name="size_batch"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Textarea
                    placeholder="Number of samples per batch"
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
            <Button type="submit">Next Iteration</Button>
          </div>
        </form>
      </Form>
    </nav>
    );
}
