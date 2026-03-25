# run this script with the following commands
# nohup ./run.sh 0 > 0.out &
# nohup ./run.sh 1 > 1.out &
# nohup ./run.sh 2 > 2.out &
# nohup ./run.sh 3 > 3.out &
# nohup ./run.sh 4 > 4.out &
# nohup ./run.sh 5 > 5.out &
# nohup ./run.sh 6 > 6.out &
# nohup ./run.sh 7 > 7.out &


if [ $1 -eq "0" ]; then
python screen.py 8 2112 7168 $1
python screen.py 8 3072 1536 $1
fi

if [ $1 -eq "1" ]; then
python screen.py 16 2112 7168 $1
python screen.py 16 3072 1536 $1
fi

if [ $1 -eq "2" ]; then
python screen.py 32 2112 7168 $1
python screen.py 32 3072 1536 $1
fi

if [ $1 -eq "3" ]; then
python screen.py 64 2112 7168 $1
python screen.py 64 3072 1536 $1
fi

if [ $1 -eq "4" ]; then
python screen.py 128 2112 7168 $1
python screen.py 128 3072 1536 $1
fi

if [ $1 -eq "5" ]; then
python screen.py 256 2112 7168 $1
python screen.py 256 3072 1536 $1
fi

if [ $1 -eq "6" ]; then
python screen.py 16384 2112 7168 $1
python screen.py 16384 3072 1536 $1
fi

# if [ $1 -eq "7" ]; then
# python screen.py 64 2112 7168 $1
# fi

