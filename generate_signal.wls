#!/usr/bin/env wolframscript
(* ::Package:: *)

file = ToFileName[{$CommandLine[[3]]}]
Print[file]
parameters = Import[file, {"Data", 1}];
frequencies = parameters[[All, 1]];
period = parameters[[All, 2]];
signs = parameters[[All, 3]];
trig = parameters[[All, 4]];
ps = parameters[[All, 5]];
intsgns = parameters[[All,6]];
funcs= Table[intsgns[[i]]*Cos[Integrate[frequencies[[i]] + signs[[i]]*Cos[x*(period[[i]] + 0.05) + ps[[i]]], x]],{i, 2, Length[frequencies]} ];
sig[t_]:= Total[funcs /. x -> {t}];
sDiscrete = ParallelTable[sig[x], {x, -500, 500, 1/360}];
