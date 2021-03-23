#!/usr/bin/env gnuplot
set terminal pngcairo size 800,480 font 'Ubuntu,14'
set output 'negative_binomial.png'


set xr [1:500]
set yr [0:0.1]
set ytics format "%5.2f"

set xlabel "Count: k"
set ylabel "Probability Density Function"

set log x
set sample 1e3

poisspdf(x,l) = l**x/gamma(x+1)*exp(-l)
lnbinpdf(k,r,p) = \
  lgamma(k+r)-lgamma(r)-lgamma(k+1)+r*log(p)+k*log(1-p)
nbinpdf(k,r,p) = exp(lnbinpdf(k,r,p))
p(r,m) = r/(r+m)

plot poisspdf(x,20.) lw 2 t "Poisson Dist. ({/Symbol l}=20)", \
     for [m=1:10:2] (r=2**(m-1), nbinpdf(x,r,p(r,20.))) \
     lw 1.8 lc 7 dt 1+(m-1)/2 \
     t sprintf("NegBin(2^{%d},%.2f)", m-1, p(2.0**(m-1),20.))
