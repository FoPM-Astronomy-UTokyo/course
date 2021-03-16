#!/usr/bin/gnuplot
set terminal pngcairo size 600,400 font 'Ubuntu,12'
set output 'inverse.png'


set xr [-5:5]
set yr [-0.3:1]
set xlabel "Random variable: x'"
set ylabel "Cumulative PDF C(x) / Random variable: u"
set ytics 0,0.2,1 format '%4.1f'
set xzeroaxis ls 1
set key above right Right

set sample 1e3
f(x) = (1+erf(x/sqrt(2)))/2.0
invf(x) = inverf(2*x-1.0)*sqrt(2)
step(x,a) = (x<invf(a))?(a):(-1.0)

plot for [n=0:30] step(x,n/30.0) lc 2 not, \
     f(x) t "Cumulative PDF: C(x)" ls 1 lc 1, \
     0.25*exp(-x**2/2.0)-0.3 lc 7 lw 2 not
