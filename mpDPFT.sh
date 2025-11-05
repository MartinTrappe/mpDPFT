#!/bin/bash

#sysbench cpu --cpu-max-prime=20000 --threads=$(nproc) run (check "total number of events" --- CQT_laptop->158189)
#execute in command line, see README.md

shopt -s nullglob

threads=${1:-$(nproc)}
job=$2

DEFAULT_DIR="$HOME/Desktop/PostDoc/Code/mpDPFT" # Intended default directory
if [ -d "$DEFAULT_DIR" ]; then
    ProgramDirectory="$DEFAULT_DIR"
    cp *.sh $ProgramDirectory
else
    ProgramDirectory="$(pwd)" # Fallback: absolute path of the current working directory (location from where mpDPFT.sh is called)
fi
ProgramDirectory="$(cd "$ProgramDirectory" && pwd)" # Normalize the path
echo "Using ProgramDirectory: $ProgramDirectory"

cd $ProgramDirectory
rm -f -- *.pdf
rm -f -- *.eps
rm -rf -- epslatex2epspdf_tmp
rm -f -- tmp_split.*
rm -f -- *.backup
rm -f -- *.new
rm -f -- *.*.new
rm -f -- *Movie*
rm -f -- mpDPFT_OPLenergies.dat
rm -f -- mpDPFT_Den_Cube.dat
chmod u+rwx "$ProgramDirectory"
chmod -R u+rwx "$ProgramDirectory/Eigen_Headers"
chmod -R u+rwx "$ProgramDirectory/CEC2014_input_data"
chmod -R u+rwx "$ProgramDirectory/mpScripts"
cp $ProgramDirectory/mpDPFT.input $ProgramDirectory/mpDPFT.tmpinput
cp $ProgramDirectory/mpDPFT.input $ProgramDirectory/mpDPFT.originput
FILE=$ProgramDirectory/run$job
if [ -d "$FILE" ];
    then echo "job directory $FILE exists"
    else
        mkdir run$job
        echo "job directory $FILE created"
fi
cd $ProgramDirectory/run$job/
rm -f -- TabFunc_X2C_*.dat
rm -f -- TabFunc_QuadraticProgram*.dat
rm -f -- TabFunc_K*GoodTriangles*.dat
rm -f -- TabFunc_NYFunction*.dat
rm -f -- TabFunc_Nuclei*.dat
rm -f -- mpDPFT_AuxMat*.dat
cd $ProgramDirectory
cp *.*input $ProgramDirectory/run$job
find "$ProgramDirectory" -maxdepth 1 -type f -exec cp -up -t "$ProgramDirectory/run$job" {} +
cp -up -r $ProgramDirectory/Eigen_Headers $ProgramDirectory/run$job
cp -up -r $ProgramDirectory/CEC2014_input_data $ProgramDirectory/run$job
cp -up -r $ProgramDirectory/mpScripts $ProgramDirectory/run$job
cd $ProgramDirectory/run$job/
rm -f -- *.pdf
rm -f -- *.eps
rm -rf -- epslatex2epspdf_tmp
rm -f -- tmp_split.*
rm -f -- *.backup
rm -f -- *.new
rm -f -- *.*.new
rm -f -- *Movie*
rm -f -- mpDPFT_DynDFTe_*.dat
rm -f -- mpDPFT_OPLenergies.dat
rm -f -- mpDPFT_RBF_*.dat
rm -f -- mpDPFT_1pExDFT_MonitorMatrix_*.dat
rm -f -- mpDPFT_ObjFunc*.*
rm -f -- mpDPFT_TabFunc_NYFunction*.*
rm -f -- mpDPFT_testK*.*
rm -f -- $ProgramDirectory/run$job/TabFunc_Hint.dat
rm -f -- "$ProgramDirectory/run$job/TabFunc_Hint.dat"
hint=( "$ProgramDirectory/run$job"/TabFunc_Hint*.dat )
((${#hint[@]})) && mv -- "${hint[0]}" "$ProgramDirectory/run$job/TabFunc_Hint.dat"
VInterpolIdentifier="mpDPFT_V_*.dat" && VInterpolIdentifier=$(echo $VInterpolIdentifier| cut -c 10-24) && if [[ ${#VInterpolIdentifier} -lt 15 ]]; then VInterpolIdentifier="?"; fi && echo "$VInterpolIdentifier" > mpDPFT_Aux.dat && mv mpDPFT_V_*.dat mpDPFT_V.dat
#make clean
make -j$(nproc)
export OMP_NUM_THREADS=$threads
export OMP_THREAD_LIMIT=$threads
# export OMP_MAX_ACTIVE_LEVELS=$threads
# export OMP_NESTED=true
# export SUNW_MP_MAX_POOL_THREADS=$threads-1
# export SUNW_MP_MAX_NESTED_LEVELS=2
# export SUNW_MP_MAX_ACTIVE_LEVELS=2
#export OMP_SCHEDULE=OMP_SCHED_STATIC
#export OMP_SCHEDULE=omp_sched_dynamic
#export OMP_SCHEDULE=OMP_SCHED_GUIDED
#export OMP_SCHEDULE=OMP_SCHED_AUTO
#export ASAN_OPTIONS=detect_leaks=1:print_summary=1:verbosity=1
nice -19 ./mpDPFT #option 1; default
#echo "gdb..." && export OMP_NUM_THREADS=1 && gdb mpDPFT #option 2; for debugging. Select in MakeFile: CC= g++ -ggdb3 ...; type 'run' when in (gdb) terminal, type 'bt' or 'thread apply all bt' for back-tracing; when stuck -> ctrl-c -> bt -> cont
#echo "valgrind..." && valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./mpDPFT
#echo "valgrind..." && valgrind --tool=massif --massif-out-file=massif.out ./mpDPFT
chmod u+rwx *.sh
chmod u+rwx *.* *
FILE=$ProgramDirectory/run$job/mpDPFT_MovieData.tmp
if test -f "$FILE"; then
    mkdir Movie
    cp mpDPFT_MovieData.tmp mpDPFT_MovieData.dat
    cp mpDPFT_Movie.sh mpDPFT_MovieData.dat $ProgramDirectory/run$job/Movie/
    cd $ProgramDirectory/run$job/Movie/
    ./mpDPFT_Movie.sh
    cp mpDPFT_Movie.mp4 $ProgramDirectory/run$job/
    cd $ProgramDirectory/run$job/
    rm -R $ProgramDirectory/run$job/Movie/
fi
FILE2=$ProgramDirectory/run$job/mpDPFT_OPLenergies.dat
if test -f "$FILE2"; then
    ./mpDPFT_OPLplots.sh
fi
./mpDPFT_Plots.sh
rm -f -- *.eps
chmod u+rwx mpDPFT_CombinedPlots.tex
pdflatex mpDPFT_CombinedPlots.tex
rm -f -- mpDPFT_CombinedPlots.log
rm -f -- mpDPFT_CombinedPlots.aux
rm -f -- texput.log
read -r VInterpolIdentifier < "mpDPFT_Aux.dat"
echo "VInterpolIdentifier=$VInterpolIdentifier" && rm -f -- mpDPFT_Aux.dat && TimeStamp="$(date +%Y%m%d_%H%M%S)" && DirectoryName="mpDPFT_$TimeStamp-$VInterpolIdentifier"
mkdir $ProgramDirectory/#DATA/#zips/$DirectoryName/
cp -r *.cpp *.h *.hpp *.*input Makefile README.md *.sh *.tex epslatex2epspdf *.info *.dat *.pdf *.mp4 *.sty Eigen_Headers mpScripts CEC2014_input_data $ProgramDirectory/#DATA/#zips/$DirectoryName/
cd $ProgramDirectory/#DATA/#zips/$DirectoryName/
mv mpDPFT.originput mpDPFT.input && mv mpDPFT_V.dat mpDPFT_V_$TimeStamp.dat
echo "confidentiality issue: files in mpScripts/Project_ItaiArad_MIT are not added to the #Source_Backups"
zip -r mpDPFT_SOURCE_$TimeStamp-$VInterpolIdentifier.zip *.cpp *.h *.hpp *.input *.sty TabFunc*.dat epslatex2epspdf Makefile README.md mpDPFT.sh mpDPFTmanualOPTloopBreakQ.dat mpDPFTmanualSCloopBreakQ.dat Eigen_Headers CEC2014_input_data -x "mpScripts/Project_ItaiArad_MIT/*" && chmod u+rwx *.zip
cp mpDPFT_SOURCE_*.zip $ProgramDirectory/#Source_Backups
rm -f -- *.zip
cd $ProgramDirectory/run$job/
if test -f "$FILE"; then
    rm -f -- mpDPFT_MovieData.tmp
    #okular mpDPFT_CombinedPlots.pdf & xdg-open mpDPFT_Movie.mp4
    #okular mpDPFT_CombinedPlots.pdf & vlc mpDPFT_Movie.mp4
fi
okular mpDPFT_CombinedPlots.pdf
