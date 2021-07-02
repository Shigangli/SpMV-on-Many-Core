#pragma OPENCL EXTENSION cl_khr_fp64:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable

__kernel void res_accumulate(__global double* inter_res,__global double *res,int len,int reslen,int times)
{
    unsigned int tid = get_global_id(0);
    unsigned int tnum= get_local_size(0)*get_num_groups(0);
    double sum=0.0;
    for(int i=0;i<times;i++){
        if(tid<len){
            sum+=inter_res[i*len+tid];
        }
    }
    if(tid<reslen)
        res[tid]=sum;
}
