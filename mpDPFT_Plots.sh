
#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2DcontourDen0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Den0'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [-0.0160495:0.919354] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:4 with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-CutDen0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Den0 along (-8,0)\$\to\$(8,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [-0.016050:0.919354]
mu(x)=8.593698
plot 'mpDPFT_CutData.dat' using 1:2 with lines ls 1 title 'Den0'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2DcontourEnv0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Env0'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [1.77494e-31:70.4] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:5 with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-CutEnv+V0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Env+V0 along (-8,0)\$\to\$(8,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [0.000000:35.2]
mu(x)=8.593698
plot 'mpDPFT_CutData.dat' using 1:3 with lines ls 3 title 'Env0', mu(x) ls 2 title '\$\mu\$', 'mpDPFT_CutData.dat' using 1:4 with lines ls 7 title 'V0' 
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2DcontourV0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'V0'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [1.82039:70.3955] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:6 with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-LogPlot-CutDen0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'LogPlotDen0 along (-8,0)\$\to\$(8,0)'
set logscale y
set format y "%e"
set samples 1000
set yrange [1e-16:0.919354]
mu(x)=8.593698
plot 'mpDPFT_CutData.dat' using 1:2 with lines ls 1 title 'Den0'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2DcontourDen1/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Den1'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [-0.022946:0.856896] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:7 with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-CutDen1/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Den1 along (-8,0)\$\to\$(8,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [-0.022946:0.856896]
mu(x)=8.598694
plot 'mpDPFT_CutData.dat' using 1:5 with lines ls 1 title 'Den1'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2DcontourEnv1/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Env1'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [1.77494e-31:70.4] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:8 with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-CutEnv+V1/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Env+V1 along (-8,0)\$\to\$(8,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [0.000000:35.2]
mu(x)=8.598694
plot 'mpDPFT_CutData.dat' using 1:6 with lines ls 3 title 'Env1', mu(x) ls 2 title '\$\mu\$', 'mpDPFT_CutData.dat' using 1:7 with lines ls 7 title 'V1' 
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2DcontourV1/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'V1'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [2.00504:70.3958] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:9 with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-LogPlot-CutDen1/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'LogPlotDen1 along (-8,0)\$\to\$(8,0)'
set logscale y
set format y "%e"
set samples 1000
set yrange [1e-16:0.856896]
mu(x)=8.598694
plot 'mpDPFT_CutData.dat' using 1:5 with lines ls 1 title 'Den1'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2Dcontour-TotalDensity/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'TotalDensity'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [-0.0215543:1.45961] noreverse nowriteback
set pm3d implicit
set palette defined ( 0 "black", 0.05 "blue", 0.3 "cyan", 0.45 "green", 0.6 "yellow", 0.8 "orange", 1 "red" )
splot 'mpDPFT_ContourData.dat' using 1:2:((\$4)+(\$7)) with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-2Dcontour-DensityDifference/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,10
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'DensityDifference'
set view map
unset surface
set style data pm3d
set xrange [-8:8] noreverse nowriteback
set yrange [-8:8] noreverse nowriteback
set cbrange [-0.589666:0.589666] noreverse nowriteback
set pm3d implicit
set palette defined ( -0.05 "black", 0.05 "#0000dd", 0.1 "#0000ff", 0.2 "#00ddff", 0.45 "#ff88ff", 0.5 "white", 0.55 "#88ff88", 0.8 "yellow", 0.9 "orange", 0.95 "#ff6633", 1.05 "#cc0000")
splot 'mpDPFT_ContourData.dat' using 1:2:((\$4)-(\$7)) with pm3d notitle
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-Overview-CutDen/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set style line 16 dt (4,1) lw 6 lc rgb "#ffffff"
set title 'Overview Den along (-8,0)\$\to\$(8,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [-0.022946:0.919354]
plot 'mpDPFT_CutData.dat' using 1:2 with lines ls 1 title 'Den0', 'mpDPFT_CutData.dat' using 1:5 with lines ls 2 title 'Den1'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo
