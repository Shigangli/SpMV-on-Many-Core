#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <limits>
#include <vector>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define MAX_SOURCE_SIZE (0x100000)

#define PERF

using namespace std;

typedef struct clContext{
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_platform_id platform_id;
    cl_uint num_devices;
    cl_uint num_platforms;
    vector< pair<string,cl_program *> > program;   
}clContext;

typedef struct Plan{
    int localthread;
    int cta;
    int col_delta;
    int workgroup;
    int registergroup;
    int localmemgroup;
    int block_width;
    int block_height;
    int trans;
    int tx;
    int coalesced;
    int slices;
    int bitwidth;
    int dimwidth;
    int dynamic_task;
}Plan;

typedef struct TimeRecord{
   cl_long kerneltime,totaltime;
   cl_long min_kerneltime, min_totaltime;
   cl_long min_cputime;
}TimeRcd;

void getClContext(clContext *clCxt);
void releaseContext(clContext *clCxt);
cl_kernel getKernel(string source,string kernelName, vector< pair<size_t,const void *> > args,
                         char * build_options,clContext *clCxt);
cl_program getProgram(string source,char * build_options,clContext *clCxt);
void executeKernel(vector<pair<size_t * ,size_t *> > &threads, cl_kernel *kernel,clContext *clCxt,int times=1,int record=1);
void create(clContext *clCxt, cl_mem *mem, int len);
void upload(clContext *clCxt,void *data,cl_mem &gdata,int data_size);
void download(clContext *clCxt,cl_mem &gdata,void *data,int data_size);
