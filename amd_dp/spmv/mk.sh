rm test -f
g++ -w -g  -o test -I/opt/AMDAPPSDK/include ../init.cpp test.cpp  -L/opt/AMDAPPSDK/lib/x86/sdk/ -lOpenCL -O3
#g++ -w -g  -o test -I/opt/AMDAPPSDK/include test.cpp  -L/usr/lib
