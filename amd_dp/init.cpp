#include "init.hpp"
TimeRcd timeRcd;
void getClContext(clContext *clCxt)
{
    cl_int ret;
    cl_uint num_platforms;
    ret = clGetPlatformIDs(0, NULL, &num_platforms);
    if(ret != CL_SUCCESS){
        cout << "Failed to get platform number. error code:"<< ret << endl;
        return ;
    }
    clCxt->num_platforms = num_platforms;
    cl_platform_id *platforms = (cl_platform_id *)malloc(clCxt->num_platforms*sizeof(cl_platform_id));
    ret = clGetPlatformIDs(clCxt->num_platforms,platforms,NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to get platform ID. error code:"<< ret << endl;
        return ;
    }
    clCxt->platform_id = platforms[0];
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM,(cl_context_properties) (clCxt->platform_id), 0};
    cl_context_properties *cprops = (NULL == clCxt->platform_id) ? NULL :cps;
          //create OpenCL context
    clCxt->context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        cout << "Failed to create context. error code:"<< ret << endl;
        return;
    }
    //get device id from context
    size_t device_num;
    ret = clGetContextInfo(clCxt->context, CL_CONTEXT_DEVICES, 0, NULL, &device_num);
    if(ret != CL_SUCCESS)
    {
        cout << "Failed to get device number. error code:"<< ret << endl;
        return ;
    }
    cl_device_id *devices=(cl_device_id *) malloc(device_num);
    ret = clGetContextInfo(clCxt->context, CL_CONTEXT_DEVICES, device_num, devices, NULL);
    if(ret != CL_SUCCESS)
    {
        cout << "Failed to get device ID. error code:"<< ret << endl;
        return ;
    }
    clCxt->device_id = devices[1];
    clCxt->num_devices = device_num;
    clCxt->command_queue = clCreateCommandQueue(clCxt->context, clCxt->device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    if(ret != CL_SUCCESS){
        cout << "Failed to create command queue. error code:"<< ret << endl;
        return ;
    }
}

void releaseContext(clContext *clCxt)
{
    cl_int ret;
    ret = clReleaseCommandQueue(clCxt->command_queue);
    if(ret != CL_SUCCESS){
        cout << "Failed to release command queue. error code:"<< ret << endl;
        return ;
    }
    ret = clReleaseContext(clCxt->context);
    if(ret != CL_SUCCESS){
        cout << "Failed to release context. error code:"<< ret << endl;
        return ;
    }
    for(int i = 0;i < clCxt->program.size();i ++)
    {
        ret = clReleaseProgram(*(clCxt->program[i].second));
        if(ret != CL_SUCCESS){
            cout << "Failed to release program. error code:"<< ret << endl;
            return ;
        }
    }
}

int savebinary(cl_program &program, const char *fileName)
{
    size_t binarySize;
    cl_int ret = clGetProgramInfo(program,
                            CL_PROGRAM_BINARY_SIZES,
                            sizeof(size_t),
                            &binarySize, NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to get binary size. error code:"<< ret << endl;
        return 0;
    }
    char* binary = (char*)malloc(binarySize);
    ret = clGetProgramInfo(program,
                           CL_PROGRAM_BINARIES,
                           sizeof(char *),
                           &binary,
                           NULL);
    if(ret != CL_SUCCESS){
        cout << "Failed to get binary. error code:"<< ret << endl;
        return 0;
    }

    FILE *fp = fopen(fileName, "wb+");
    if(fp != NULL)
    {
        fwrite(binary, binarySize, 1, fp);
        fclose(fp);
    }
    free(binary);
    return 1;
}

cl_program getProgram(string source,char * build_options,clContext *clCxt)
{
    cl_program *program = NULL;
    cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;
    source="kernel/"+source;
    fp = fopen(source.c_str(), "r");
    if (!fp) {
        printf("Failed to load kernel file.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    string build_options_str(build_options);
    for(int i=0;i<clCxt->program.size();i++)
    {
        if(build_options_str==clCxt->program[i].first)
            program = clCxt->program[i].second;
    }
    if(program == NULL)
    {
        char all_build_options[1024];
        string filename;
        memset(all_build_options, 0, 1024);
        all_build_options[0]='k';
        if(build_options != NULL){
            for(int i=0,j=1;build_options[i]!='\0';i++){
                if((build_options[i]<='9'&&build_options[i]>='0')||build_options[i]=='-'){
                    all_build_options[j] = build_options[i];
                    j++;
                }
            }
        }
        string binpath = "clbin/";
        if(all_build_options != NULL)
        {
            filename = binpath + all_build_options + ".clb";
        }
        else
        {
            filename = binpath + source + "_" + ".clb";
        }
        FILE *fp = fopen(filename.c_str(), "rb");
        if(fp == NULL)
        {
            program = (cl_program *)malloc(sizeof(cl_program));
            *program = clCreateProgramWithSource(clCxt->context, 1, (const char **)&source_str,
                                                            (const size_t *)&source_size, &ret);
            if(ret != CL_SUCCESS){
                cout << "Failed to create program with source. error code:"<< ret << endl;
                return NULL;
            }
            ret = clBuildProgram(*program,1,&(clCxt->device_id), build_options, NULL,NULL);
            if(ret != CL_SUCCESS){
                cout << "Failed to build program with source. error code:"<< ret << endl;
            }
            savebinary(*program, filename.c_str());
        }
        else
        {
            fseek(fp, 0, SEEK_END);
            size_t binarySize = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *binary = new char[binarySize];
            if(1 != fread(binary, binarySize, 1, fp)){
                cout << "Failed to load binary file. error code:"<< ret << endl;
                return NULL;
            }
            fclose(fp);
            program = (cl_program *)malloc(sizeof(cl_program));
            *program = clCreateProgramWithBinary(clCxt->context,
                                1,
                                &(clCxt->device_id),
                                (const size_t *)&binarySize,
                                (const unsigned char **)&binary,
                                NULL,
                                &ret);
            if(ret != CL_SUCCESS){
                cout << "Failed to create Program with binary. error code:"<< ret << endl;
                return NULL;
            }
            ret = clBuildProgram(*program, 1, &(clCxt->device_id), build_options, NULL, NULL);
            delete[] binary;
            if(ret != CL_SUCCESS){
                cout << "Failed to build program with binary. error code:"<< ret << endl;
                return NULL;
            }
        }
        //printf("Compile the source code.\n");
        char *buildLog = NULL;
        size_t buildLogSize = 0;
        clGetProgramBuildInfo(*program,clCxt->device_id,CL_PROGRAM_BUILD_LOG,buildLogSize,
                                          buildLog,&buildLogSize);
        buildLog = new char[buildLogSize];
        memset(buildLog,0,buildLogSize);
        clGetProgramBuildInfo(*program,clCxt->device_id,
                              CL_PROGRAM_BUILD_LOG,buildLogSize,buildLog,NULL);
        if(ret != CL_SUCCESS){
            cout << "\n\t\t\tBUILD LOG\n";
            cout << buildLog << endl;
            return NULL;
        }
        delete buildLog;
        clCxt->program.push_back( make_pair( build_options_str , program ));
    }
    free(source_str);
    return *program;
}



cl_kernel getKernel(string source,string kernelName, vector< pair<size_t,const void *> > args,
                    char * build_options,clContext *clCxt)
{
    cl_program *program = NULL;
    cl_kernel kernel = NULL;
    cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;
    source="kernel/"+source;
    fp = fopen(source.c_str(), "r");
    if (!fp) {
        cout << kernelName <<"  ";
        cout << "Failed to load kernel file."<<endl;
        return NULL;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    string build_options_str(build_options);
    for(int i=0;i<clCxt->program.size();i++)
    {
        if(build_options_str==clCxt->program[i].first)
            program = clCxt->program[i].second;
    }
    if(program == NULL)
    {
        char all_build_options[1024];
        string filename;
        memset(all_build_options, 0, 1024);
        all_build_options[0]='k';
        if(build_options != NULL){
            for(int i=0,j=1;build_options[i]!='\0';i++){
                if((build_options[i]<='9'&&build_options[i]>='0')||build_options[i]=='-'){
                    all_build_options[j] = build_options[i];
                   j++;
                }
            }
        }
        string binpath = "clbin/";
        if(all_build_options != NULL)
        {
            filename = binpath + all_build_options + ".clb";
        }
        else
        {
            filename = binpath + source + "_" + ".clb";
        }
        FILE *fp = fopen(filename.c_str(), "rb");
        if(fp == NULL)
        {
            program = (cl_program *)malloc(sizeof(cl_program));
            *program = clCreateProgramWithSource(clCxt->context, 1, (const char **)&source_str,
                                                            (const size_t *)&source_size, &ret);
            if(ret != CL_SUCCESS){
                cout << kernelName <<"  ";
                cout << "Failed to create program with source. error code:"<< ret << endl;
                return NULL;
            }
            ret = clBuildProgram(*program,1,&(clCxt->device_id), build_options, NULL,NULL);
            if(ret != CL_SUCCESS){
                cout << kernelName <<"  ";
                cout << "Failed to build program with source. error code:"<< ret << endl;
            }
            savebinary(*program, filename.c_str());
        }
        else
        {
            fseek(fp, 0, SEEK_END);
            size_t binarySize = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *binary = new char[binarySize];
            if(1 != fread(binary, binarySize, 1, fp)){
                cout << kernelName <<"  ";
                cout << "Failed to load binary file. error code:"<< ret << endl;
                return NULL;
            }
            fclose(fp);
            program = (cl_program *)malloc(sizeof(cl_program));
            *program = clCreateProgramWithBinary(clCxt->context,
                                1,
                                &(clCxt->device_id),
                                (const size_t *)&binarySize,
                                (const unsigned char **)&binary,
                                NULL,
                                &ret);
            if(ret != CL_SUCCESS){
                cout << kernelName <<"  ";
                cout << "Failed to create Program with binary. error code:"<< ret << endl;
                return NULL;
            }
            ret = clBuildProgram(*program, 1, &(clCxt->device_id), build_options, NULL, NULL);
            delete[] binary;
            if(ret != CL_SUCCESS){
                cout << kernelName <<"  ";
                cout << "Failed to build program with binary. error code:"<< ret << endl;
                return NULL;
            }
        }
        //printf("Compile the source code.\n");
        char *buildLog = NULL;
        size_t buildLogSize = 0;
        clGetProgramBuildInfo(*program,clCxt->device_id,CL_PROGRAM_BUILD_LOG,buildLogSize,
                                          buildLog,&buildLogSize);
        buildLog = new char[buildLogSize];
        memset(buildLog,0,buildLogSize);
        clGetProgramBuildInfo(*program,clCxt->device_id,
                              CL_PROGRAM_BUILD_LOG,buildLogSize,buildLog,NULL);
        if(ret != CL_SUCCESS){
            cout << "\n\t\t\tBUILD LOG\n";
            cout << buildLog << endl;
            return NULL;
        }
        delete buildLog;
        clCxt->program.push_back( make_pair( build_options_str , program ));
    }
    kernel = clCreateKernel(*program, kernelName.c_str(), &ret);
    if(ret != CL_SUCCESS){
        cout << kernelName <<"  ";
        cout << "Failed to create Kernel. error code:"<< ret <<"  "<<kernelName<< endl;
        return NULL;
    }
    for(int i = 0;i < args.size();i ++)
    {
        ret = clSetKernelArg(kernel,i,args[i].first,args[i].second);
        if(ret != CL_SUCCESS){
            cout << kernelName <<"  ";
            cout << "Failed to set Arg.Arg:"<< i <<", error code:"<< ret  << endl;
            return NULL;
        }
    }
    free(source_str);
    return kernel;
}

void executeKernel(vector<pair<size_t * ,size_t *> > &threads, cl_kernel *kernel,clContext *clCxt,int times,int record)
{
    cl_ulong queued,start,end;
    cl_int ret;
    if(record==1){
        for(int j=0;j<threads.size();j++){
            ret = clEnqueueNDRangeKernel(clCxt->command_queue,kernel[j],2,NULL,threads[j].first,
                                         threads[j].second,0,NULL,NULL);
            if(ret != CL_SUCCESS){
                cout << "Failed to EnqueueNDRangeKernel. error code:"<< ret  << endl;
                return ;
            }
        }
    }
    cl_ulong kernelRealExecTimeNs = 0;
    for(int i=0;i<times;i++){
        for(int j=0;j<threads.size();j++){
            if(record==1){
                cl_event time_event;
                cl_ulong queued,start,end;
                ret = clEnqueueNDRangeKernel(clCxt->command_queue,kernel[j],2,NULL,threads[j].first,
                                         threads[j].second,0,NULL,&time_event);
                clWaitForEvents(1,&time_event);
                clFinish(clCxt->command_queue);
                if(ret != CL_SUCCESS){
                    cout << "Failed to EnqueueNDRangeKernel. error code:"<< ret  << endl;
                    return ;
                }
                ret = clGetEventProfilingInfo(time_event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong) ,&start,NULL);
                if(ret != CL_SUCCESS){
                    cout << "Failed to clGetEventProfilingInfo 2. error code:"<< ret  << endl;
                }
                ret = clGetEventProfilingInfo(time_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong) ,&end,NULL);
                if(ret != CL_SUCCESS){
                    cout << "Failed to clGetEventProfilingInfo 3. error code:"<< ret  << endl;
                }
                kernelRealExecTimeNs =kernelRealExecTimeNs + end - start;
            }
            else{
                ret = clEnqueueNDRangeKernel(clCxt->command_queue,kernel[j],2,NULL,threads[j].first,
                                         threads[j].second,0,NULL,NULL);
                if(ret != CL_SUCCESS){
                    cout << "Failed to EnqueueNDRangeKernel. error code:"<< ret  << endl;
                    return ;
                }
            }
        }
    }
    if(record==1){
        clFinish(clCxt->command_queue);
        kernelRealExecTimeNs /= 1000;
        if(ret == CL_SUCCESS)
            timeRcd.totaltime = kernelRealExecTimeNs;
        else
            timeRcd.totaltime = 0;
#if defined PERF
        printf("kernel execute time=%lf  min kernel execute time=%lf  \n",
                (double)kernelRealExecTimeNs/(1000*times),(double)timeRcd.min_totaltime/(1000*times));
#endif
    }
    return ;
}

void create(clContext *clCxt, cl_mem *mem, int len)
{
    cl_int ret;
    *mem = clCreateBuffer(clCxt->context,CL_MEM_READ_WRITE,len,NULL,&ret);
    if(ret != CL_SUCCESS){
        cout << "Failed to create buffer on GPU. "<< ret << endl;
        return ;
    }
}

void upload(clContext *clCxt,void *data,cl_mem &gdata,int datalen)
{
    //write data to buffer
    cl_int ret;
    ret = clEnqueueWriteBuffer(clCxt->command_queue,gdata,CL_TRUE,0,datalen,(void *)data,0,NULL,NULL);
    if(ret != CL_SUCCESS){
        cout << "clEnqueueWriteBuffer failed." << ret<<endl;
        return ;
    }
    clFinish(clCxt->command_queue);
}

void download(clContext *clCxt,cl_mem &gdata,void *data,int data_len)
{
    cl_int ret;
    ret = clEnqueueReadBuffer(clCxt->command_queue, gdata, CL_TRUE, 0, data_len,(void *)data, 0, NULL,NULL);
    if(ret != CL_SUCCESS){
        cout << "clEnqueueReadBuffer failed. error code:" << ret <<endl;
        return ;
    }
    clFinish(clCxt->command_queue);
}

