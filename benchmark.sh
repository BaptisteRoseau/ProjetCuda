SIZES_SLOW='500 5000 10000 25000 50000'
SIZES_FAST='500 5000 10000 25000 50000 100000 500000'

BENCH_FILE_TIME=time.csv
BENCH_FILE_FLOPS=flops.csv

# Removing previous time and execution
rm -f $BENCH_FILE_TIME $BENCH_FILE_FLOPS

# Compiling source
cd nbody && make all && cd ..

# Header
echo "Nbody execution time,Size,Time(s),log,log"   > $BENCH_FILE_TIME
echo "Nbody execution GFlop/s,Size,GFlop/s,log,log" > $BENCH_FILE_FLOPS
echo ""
echo "Nbody benchmark wrote into:"
echo "   $(pwd)/$BENCH_FILE_TIME"
echo "   $(pwd)/$BENCH_FILE_FLOPS"

# Benchmark for slow binaries
bench_slow (){
    BIN=$1
    echo "$BIN"  >> $BENCH_FILE_TIME
    echo "$BIN"  >> $BENCH_FILE_FLOPS
    echo "Executing benchmark for $BIN...";
    for size in $SIZES_SLOW; do \
        echo "Size $size..." ; \
        start=`echo $(($(date +%s)))`; \
        echo "$size, $(./nbody/$BIN $size |\
        grep Average | awk -F " " '{print $4}')" 1>> $BENCH_FILE_FLOPS; \
        end=`echo $(($(date +%s)))`; \
        echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
    done
}

# Benchmark for fast binaries
bench_fast (){
    BIN=$1
    echo "$BIN"  >> $BENCH_FILE_TIME
    echo "$BIN"  >> $BENCH_FILE_FLOPS
    echo "Executing benchmark for $BIN...";
    for size in $SIZES_FAST; do \
        echo "Size $size..." ; \
        start=`echo $(($(date +%s)))`; \
        echo "$size, $(./nbody/$BIN $size |\
        grep Average | awk -F " " '{print $4}')" 1>> $BENCH_FILE_FLOPS; \
        end=`echo $(($(date +%s)))`; \
        echo "$size, $((end-start))" >> $BENCH_FILE_TIME; \
    done
}


# Executing benchmarks
bench_fast "nbodyacc"
bench_fast "nbodycu"
bench_fast "nbodysoacu"
bench_slow "nbody"
bench_slow "nbodyomp"
bench_slow "nbodysoa"

# Drwing results
python3 drawCurve.py $BENCH_FILE_TIME $BENCH_FILE_FLOPS
