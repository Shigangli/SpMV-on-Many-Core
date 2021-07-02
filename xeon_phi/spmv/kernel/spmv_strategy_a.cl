#pragma OPENCL EXTENSION cl_khr_fp64:enable
#define LOG_NUM_BANKS 5 
#define NUM_BANKS 32 
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)
#define CTA_SIZE 32
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable

//#if defined BIT_32
#define BIT_TYPE int
#define BIT_WIDTH 32
#define BIT_WIDTH_1 31
#define I_MAX UINT_MAX 
//#endif

//DIM = dimwidth/8 (32/8)
//#if defined DIM_4
#define DIM_TYPE unsigned int
//#endif


//#if defined (NB_VEC_4)
#define VEC_TYPE float4
#define vec_sum(r) (r.s0+r.s1+r.s2+r.s3)
//#endif

//#if defined BLOCK_HEIGHT_4
#define SEG_TYPE float4
#define RES_TYPE float4
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst.s0 = vec_sum(temp);\
    temp = data1[data_id];\
    temp = temp * rvector;\
    dst.s1 = vec_sum(temp);\
    temp = data2[data_id];\
    temp = temp * rvector;\
    dst.s2 = vec_sum(temp);\
    temp = data3[data_id];\
    temp = temp * rvector;\
    dst.s3 = vec_sum(temp);
//#endif

#define GET_VECTOR(rvector, vector, loc) rvector = vector[loc];
#define GET_COL(col,data_id,t) t = col[data_id];
#define NB_LSIZE 1

__kernel void spmv_bccoo(__global VEC_TYPE *data,
                         __global VEC_TYPE *data1,
                         __global VEC_TYPE *data2,
                         __global VEC_TYPE *data3,
                         __global DIM_TYPE *col,
//#if defined COL_COM_1
//                         __global short *col_delta,
//#endif
                         __global unsigned BIT_TYPE *bit,
                         __global int *res_entry,    //result entry for each thread.
                         __global volatile SEG_TYPE *inter,   //the buffer used for adjacent synchronization
//                         __global int *para_scan,      //a signal to specify if the segmented scan on the 
                                                     //last partial sum array of this workgroup can be removed
                         int groupnum,               //the number of the launched workgroups.

//#if defined TX_1
//                         __read_only image2d_t vector,
//#else
                         __global VEC_TYPE * __restrict vector,
//#endif
                         __global RES_TYPE *res)     //result
{
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_group_id(0);   
    unsigned int tid = gid * NB_LSIZE + lid;
    unsigned int data_id = tid*CTA_SIZE;
    //__global SEG_TYPE intermediate_sum[CTA_SIZE];
    //__global SEG_TYPE last_partial_sums[NB_LSIZE];
    __local SEG_TYPE intermediate_sum[CTA_SIZE];
    __local SEG_TYPE last_partial_sums[NB_LSIZE];
    //SEG_TYPE inter[groupnum];
    SEG_TYPE ft=0.0,ft1=0.0;
    VEC_TYPE temp,rvector;
    unsigned int t1=0,t2=0,t3=0,t4=0,thread_res_entry_location=res_entry[tid];
    //__global unsigned int block_res_entry_location;
    //__global SEG_TYPE previous_workgroups_last_partial_sum;
    __local SEG_TYPE previous_workgroups_last_partial_sum;
    __local unsigned int block_res_entry_location;
    if(lid==0) block_res_entry_location=thread_res_entry_location;
    unsigned BIT_TYPE bit_flag = bit[tid];

    //1. serial scan
    t4 = col[data_id];
    {GET_VECTOR(rvector,vector,t4)}
    GET_DATA(intermediate_sum[0])
    data_id += 1;
    for(t1=1,t2=1<<BIT_WIDTH_1;t1<CTA_SIZE;t1++,t2=t2>>1)
    {  
        {GET_COL(col,data_id,t4)}
        {GET_VECTOR(rvector,vector,t4)}
        GET_DATA(intermediate_sum[t1])
        intermediate_sum[t1] = ((bit_flag&t2)>0)*intermediate_sum[t1-1] + intermediate_sum[t1];
        data_id += 1;
    }
    last_partial_sums[lid] = ((bit_flag&1)>0)*intermediate_sum[CTA_SIZE-1];

    //2. adjacent synchronization

    if(lid==0){    //generate the information if the adjacent synchronization chain can be cut off.
        if(gid != groupnum-1){
            t1 = res_entry[tid + NB_LSIZE];
            t1 = t1 - thread_res_entry_location;
        }
        else
            t1 = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);  //0.1647ms

    if(lid==0){   //adjacent synchronization
        if(gid ==0||t1!=0){
            inter[gid] = last_partial_sums[NB_LSIZE-1];
        }
        if(gid != 0){
#if defined BLOCK_HEIGHT_1
            while((ft1=inter[gid-1])==FLT_MAX){}
#else
            while(1){
                ft1=inter[gid-1];
                if(ft1.s0!=FLT_MAX)
                  break;
            }
#endif
            inter[gid-1]=FLT_MAX;
        }
        else
            ft1=0.0; //for the first threadgroup
        if(gid!=0 && t1==0){
            inter[gid] = ft1 + last_partial_sums[NB_LSIZE-1];
        }
        previous_workgroups_last_partial_sum = ft1;   
    }
    barrier(CLK_LOCAL_MEM_FENCE);
   
   //3. write to the result array
    t3 = 1;
    for(t1=0,t2=1<<BIT_WIDTH_1;t1<CTA_SIZE;t1++,t2=t2>>1)
    {
        if((bit_flag&t2)==0){
            ft=previous_workgroups_last_partial_sum*t3 + intermediate_sum[t1];
#if defined BLOCK_HEIGHT_3
            res[thread_res_entry_location*3]=ft.s0;
            res[thread_res_entry_location*3+1]=ft.s1;
            res[thread_res_entry_location*3+2]=ft.s2;
#else
            res[thread_res_entry_location]=ft;
#endif
            t3=0; //only accumulate for the first row
            thread_res_entry_location++;
        }
    }  //0.1834ms
}
