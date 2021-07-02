// @Authors
//    Shengen Yan,yanshengen@gmail.com
/**************************************PUBLICFUNC*************************************/
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#define LOG_NUM_BANKS 5 
#define NUM_BANKS 32 
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable

#if defined BIT_8
#define BIT_TYPE char
#define BIT_WIDTH 8
#define BIT_WIDTH_1 7
#define I_MAX 255
#endif
#if defined BIT_16
#define BIT_TYPE short
#define BIT_WIDTH 16
#define BIT_WIDTH_1 15
#define I_MAX 65535
#endif
#if defined BIT_32
#define BIT_TYPE int
#define BIT_WIDTH 32
#define BIT_WIDTH_1 31
#define I_MAX UINT_MAX 
#endif
#if defined BIT_64
#define BIT_TYPE long 
#define BIT_WIDTH 64
#define BIT_WIDTH_1 63
#define I_MAX ULONG_MAX 
#endif

#if defined DIM_2
#define DIM_TYPE unsigned short
#endif
#if defined DIM_4
#define DIM_TYPE unsigned int
#endif

#if defined (NB_L64)
#define NB_LSIZE 64
#define NB_LSIZE_LOG 6 
#define H_NB_LSIZE 32
#define NB_LSIZE_1 63
#define NB_LSIZE_2 65
#define NB_LSIZE_P 1 
#if defined (NB_CTA_8)
#define NB_CTA_NUM 8 
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 4 
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 2
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 1
#endif
#endif

#if defined (NB_L128)
#define NB_LSIZE 128
#define NB_LSIZE_LOG 7 
#define H_NB_LSIZE 64
#define NB_LSIZE_1 127
#define NB_LSIZE_2 129
#define NB_LSIZE_P 3 
#if defined (NB_CTA_8)
#define NB_CTA_NUM 16
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 8
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 4
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 2
#endif
#endif

#if defined (NB_L256)
#define NB_LSIZE 256 
#define NB_LSIZE_LOG 8 
#define H_NB_LSIZE 128
#define NB_LSIZE_1 255 
#define NB_LSIZE_2 257 
#define NB_LSIZE_P 7 
#if defined (NB_CTA_8)
#define NB_CTA_NUM 32 
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 16 
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 8
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 4
#endif
#endif

#if defined (NB_L512)
#define NB_LSIZE 512 
#define NB_LSIZE_LOG 9
#define H_NB_LSIZE 256
#define NB_LSIZE_1 511 
#define NB_LSIZE_2 513 
#define NB_LSIZE_P 15 
#if defined (NB_CTA_8)
#define NB_CTA_NUM 64
#endif
#if defined (NB_CTA_16)
#define NB_CTA_NUM 32 
#endif
#if defined (NB_CTA_32)
#define NB_CTA_NUM 16
#endif
#if defined (NB_CTA_64)
#define NB_CTA_NUM 8
#endif
#endif


//#define DYNAMIC_TASK

#if defined (NB_VEC_1)
#define VEC_TYPE double
#define vec_sum(r) (r)
#endif
#if defined (NB_VEC_2)
#define VEC_TYPE double2
#define vec_sum(r) (r.s0+r.s1)
#endif
#if defined (NB_VEC_4)
#define VEC_TYPE double4
#define vec_sum(r) (r.s0+r.s1+r.s2+r.s3)
#endif
#if defined (NB_VEC_8)
#define VEC_TYPE double8
#define vec_sum(r) (r.s0+r.s1+r.s2+r.s3+r.s4+r.s5+r.s6+r.s7)
#endif
#if defined TX_1
#if defined NB_VEC_1
#define GET_VECTOR(rvector,vector,loc) \
    int2 coord; \
    coord.x = loc % TX_WIDTH; \
    coord.y = loc / TX_WIDTH; \
    double4 tvector = read_imagef(vector,smp,coord);\
    rvector = tvector.s0;
#endif
#if defined NB_VEC_2
#define GET_VECTOR(rvector,vector,loc) \
    int2 coord; \
    coord.x = loc % TX_WIDTH; \
    coord.y = loc / TX_WIDTH; \
    double4 tvector = read_imagef(vector,smp,coord);\
    rvector = tvector.lo;
#endif
#if defined NB_VEC_4
#define GET_VECTOR(rvector,vector,loc) \
    int2 coord;\
    coord.x = loc % TX_WIDTH; \
    coord.y = loc / TX_WIDTH; \
    rvector = read_imagef(vector, smp, coord);
#endif
#if defined NB_VEC_8
#define GET_VECTOR(rvector,vector,loc) \
    int2 coord; \
    coord.x = (loc<<1) % TX_WIDTH; \
    coord.y = (loc<<1) / TX_WIDTH; \
    double4 tvector = read_imagef(vector,smp,coord);\
    rvector.s0123 = tvector; \
    coord.x = ((loc<<1)+1) % TX_WIDTH; \
    coord.y = ((loc<<1)+1) / TX_WIDTH; \
    tvector = read_imagef(vector,smp,coord);\
    rvector.s4567 = tvector;
#endif
#else
#define GET_VECTOR(rvector,vector,loc) rvector = vector[loc];
#endif

#if defined COL_COM_1
#define GET_COL(col,data_id,t) \
    int tt = col_delta[data_id]; \
    if(tt!=-1) t = t + tt; \
    else t = col[data_id];
#else
#define GET_COL(col,data_id,t) t = col[data_id];
#endif

#if defined BLOCK_HEIGHT_1
#define SEG_TYPE double
#define RES_TYPE double
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst = vec_sum(temp);
#endif
#if defined BLOCK_HEIGHT_2
#define SEG_TYPE double2
#define RES_TYPE double2
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst.s0 = vec_sum(temp);\
    temp = data1[data_id];\
    temp = temp * rvector;\
    dst.s1 = vec_sum(temp);
#endif
#if defined BLOCK_HEIGHT_3
#define SEG_TYPE double4
#define RES_TYPE double
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst.s0 = vec_sum(temp);\
    temp = data1[data_id];\
    temp = temp * rvector;\
    dst.s1 = vec_sum(temp);\
    temp = data2[data_id];\
    temp = temp * rvector;\
    dst.s2 = vec_sum(temp);
#endif
#if defined BLOCK_HEIGHT_4
#define SEG_TYPE double4
#define RES_TYPE double4
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
#endif

__kernel void spmv_bccoo(__global VEC_TYPE *data,
#if defined BLOCK_HEIGHT_2
                         __global VEC_TYPE *data1,
#endif
#if defined BLOCK_HEIGHT_3
                         __global VEC_TYPE *data1,
                         __global VEC_TYPE *data2,
#endif
#if defined BLOCK_HEIGHT_4
                         __global VEC_TYPE *data1,
                         __global VEC_TYPE *data2,
                         __global VEC_TYPE *data3,
#endif
                         __global DIM_TYPE *col,
#if defined COL_COM_1
                         __global short *col_delta,
#endif
                         __global unsigned BIT_TYPE *bit,
                         __global int *res_entry,    //result entry for each thread.
                         __global volatile SEG_TYPE *inter,   //the buffer used for adjacent synchronization
                         __global int *para_scan,      //a signal to specify if the segmented scan on the 
                                                     //last partial sum array of this workgroup can be removed
                         int groupnum,               //the number of the launched workgroups.

#if defined TX_1
                         __read_only image2d_t vector,
#else
                         __global VEC_TYPE * __restrict vector,
#endif
                         __global RES_TYPE *res)     //result
{
    unsigned int lid = get_local_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
#if defined DYNAMIC_TASK
    unsigned int gid;
    __local int gid_;
    if(lid == 0)
        gid_ = atom_add((__global int*)(para_scan+groupnum),1);
    barrier(CLK_LOCAL_MEM_FENCE);
    gid = gid_;
#else
   unsigned int gid = get_group_id(0);
#endif
#if NB_LOCAL_SIZE   //if using local memory to cache the intermediate sums.
    __local SEG_TYPE intermediate_sum_l[NB_CTA_NUM][NB_LOCAL_SIZE][NB_LOCAL_SIZE];  //intermediate_sum in local memory
    unsigned int ctaid = lid / NB_LOCAL_SIZE;
    unsigned int laneid = lid & (NB_LOCAL_SIZE-1);
#endif
    __local SEG_TYPE last_partial_sums[NB_LSIZE+NB_LSIZE_P],previous_workgroups_last_partial_sum;
    __local unsigned int head_flag[2][NB_LSIZE+NB_LSIZE_P];               //head flag array for parallel segmented scan.
    __local unsigned int block_res_entry_location;   //the first result location of the whole workgroup.
    unsigned int tid = gid * NB_LSIZE + lid;
#if defined TRANS_1     //if using offline transpose.
    unsigned int data_id = tid;
#else
    unsigned int data_id = (gid * NB_CTA_NUM + ctaid)* NB_LOCAL_SIZE*(NB_LOCAL_SIZE+NB_REG_SIZE)+laneid;
#endif
#if NB_REG_SIZE     //if using register files to cache the intermediate sums.
    SEG_TYPE intermediate_sum_r[NB_REG_SIZE];
#endif
    SEG_TYPE ft=0.0,ft1=0.0;
    VEC_TYPE temp,rvector;
    unsigned int t1=0,t2=0,t3=0,t4=0,thread_res_entry_location=res_entry[tid];
    unsigned BIT_TYPE bit_flag_for_register=I_MAX,bit_flag_for_localmem=I_MAX;
    if(lid==0) block_res_entry_location=thread_res_entry_location;
#if defined TRANS_1            //if using offline transpose strategy
    t3 = groupnum * NB_LSIZE;
    t4 = col[data_id];
    ft1 =0.0;
#if NB_REG_SIZE
    bit_flag_for_register = bit[tid*(NB_REG_SIZE+NB_LOCAL_SIZE)/NB_REG_SIZE];
    {GET_VECTOR(rvector,vector,t4)}
    GET_DATA(intermediate_sum_r[0])
    data_id += t3;
    for(t1=1,t2=1<<BIT_WIDTH_1;t1<NB_REG_SIZE;t1++,t2=t2>>1){
        {GET_COL(col,data_id,t4)}
        {GET_VECTOR(rvector,vector,t4)}
        GET_DATA(intermediate_sum_r[t1])
        intermediate_sum_r[t1] = ((bit_flag_for_register&t2)>0)*intermediate_sum_r[t1-1] + intermediate_sum_r[t1];
        data_id += t3;
    }
    ft1 = ((bit_flag_for_register&t2)>0)*intermediate_sum_r[NB_REG_SIZE-1];
#endif
#if NB_LOCAL_SIZE  
    bit_flag_for_localmem = bit[tid*(NB_REG_SIZE+NB_LOCAL_SIZE)/NB_LOCAL_SIZE+(NB_REG_SIZE!=0)];
#if NB_REG_SIZE
    {GET_COL(col,data_id,t4)}
#endif
    {GET_VECTOR(rvector,vector,t4)}
    GET_DATA(ft)
    intermediate_sum_l[ctaid][laneid][0] = ft1  + ft;   //add the last partial sum in the register cache.
    data_id += t3;
    for(t1=1,t2 = 1<<BIT_WIDTH_1;t1<NB_LOCAL_SIZE;t1++,t2=t2>>1)
    {
        {GET_COL(col,data_id,t4)}
        {GET_VECTOR(rvector,vector,t4)}
        GET_DATA(ft)
        intermediate_sum_l[ctaid][laneid][t1] = 
            ((bit_flag_for_localmem&t2)>0)*intermediate_sum_l[ctaid][laneid][t1-1] + ft; 
        data_id += t3;
    }
#endif
#else              //if using online transpose strategy.
    t3=data_id;
    ft1 = 0.0;
#if NB_REG_SIZE
    bit_flag_for_register = bit[tid*(NB_REG_SIZE+NB_LOCAL_SIZE)/NB_REG_SIZE];
    for(t1=0;t1<NB_LOCAL_SIZE;t1++)
    {
        t4 = col[data_id];
        {GET_VECTOR(rvector,vector,t4)}
        GET_DATA(ft)
        intermediate_sum_l[ctaid][t1][laneid] = ft;
        data_id =data_id + NB_LOCAL_SIZE + NB_REG_SIZE;
    }
    intermediate_sum_r[0] = intermediate_sum_l[ctaid][laneid][0];
    for(t1=1,t2 = 1<<BIT_WIDTH_1;t1<NB_LOCAL_SIZE;t1++,t2=t2>>1){
        intermediate_sum_r[t1] = intermediate_sum_l[ctaid][laneid][t1] + 
            ((bit_flag_for_register&t2)>0)*intermediate_sum_r[t1-1];
    }
    ft1 = intermediate_sum_r[NB_REG_SIZE-1]*((bit_flag_for_register&t2)>0);
    data_id =t3 + NB_REG_SIZE; 
#endif
    bit_flag_for_localmem = bit[tid*(NB_REG_SIZE+NB_LOCAL_SIZE)/NB_LOCAL_SIZE+(NB_REG_SIZE!=0)];
    for(t1=0;t1<NB_LOCAL_SIZE;t1++)
    {
        t4 = col[data_id];
        {GET_VECTOR(rvector,vector,t4)}
        GET_DATA(ft)
        intermediate_sum_l[ctaid][t1][laneid] = ft;
        data_id =data_id + NB_LOCAL_SIZE + NB_REG_SIZE;
    }
    intermediate_sum_l[ctaid][laneid][0] =intermediate_sum_l[ctaid][laneid][0] + ft1;
    for(t1=1,t2 = 1<<BIT_WIDTH_1;t1<NB_LOCAL_SIZE;t1++,t2=t2>>1){
        intermediate_sum_l[ctaid][laneid][t1] =intermediate_sum_l[ctaid][laneid][t1] + 
                    ((bit_flag_for_localmem&t2)>0)*intermediate_sum_l[ctaid][laneid][t1-1];
    } 
#endif
#if NB_LOCAL_SIZE  
    ft = intermediate_sum_l[ctaid][laneid][NB_LOCAL_SIZE-1]*((bit_flag_for_localmem&1)>0);
#else
    ft = intermediate_sum_r[NB_REG_SIZE-1]*((bit_flag_for_register&1)>0);
#endif
    last_partial_sums[lid+GET_CONFLICT_OFFSET(lid)] = ft;
#if defined SEGSCAN_1        //0.1555ms
    if(para_scan[gid]==1){     //parallel segmented scan on the last partial sums.
        head_flag[0][lid+GET_CONFLICT_OFFSET(lid)]=
                       1 - ((bit_flag_for_register & bit_flag_for_localmem) == I_MAX);  //generate the head flags.
        head_flag[1][lid+GET_CONFLICT_OFFSET(lid)]=
                       1 - ((bit_flag_for_register & bit_flag_for_localmem) == I_MAX);  //backup the head flags.
        barrier(CLK_LOCAL_MEM_FENCE);
        t1 = 1;
        for(t2=H_NB_LSIZE;t2>0;t2>>=1){
            t3 = t1 * ((lid<<1)+1)-1,t4 = t3 + t1;
            t3 += GET_CONFLICT_OFFSET(t3);
            t4 += GET_CONFLICT_OFFSET(t4);
            if(lid < t2){
                if(head_flag[0][t4]==0)
                    last_partial_sums[t4]=last_partial_sums[t4]+last_partial_sums[t3]; 
                head_flag[0][t4] = head_flag[0][t4]||head_flag[0][t3];
            }
            t1<<=1;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(lid == 0){
            last_partial_sums[NB_LSIZE_1+NB_LSIZE_P] = 0.0;
        }
        int local_size = NB_LSIZE+NB_LSIZE_P;
        for(t2 = 1; t2<NB_LSIZE;t2=t2*2){
            barrier(CLK_LOCAL_MEM_FENCE);
            t1 = t1/2; 
            t3 = t1 * ((lid*2)+1)-1,t4 = t3 + t1;
            t3 += GET_CONFLICT_OFFSET(t3);
            t4 += GET_CONFLICT_OFFSET(t4);
            if(lid < t2){
                int kkk=t1 * ((lid<<1)+1)+GET_CONFLICT_OFFSET(t1 * ((lid<<1)+1));
                t3 = t3>=local_size?local_size-1:t3;
                t4 = t4>=local_size?local_size-1:t4;
                kkk = kkk>=local_size?local_size-1:kkk;
                ft1 = last_partial_sums[t3];
                last_partial_sums[t3] = last_partial_sums[t4];
	        if(head_flag[1][kkk] == 1){
                    ft1 = 0.0;
                }
                if(head_flag[1][kkk] != 1){
                  if( head_flag[0][t3]!=1)
                    ft1 = last_partial_sums[t4]+ft1;
                }
                last_partial_sums[t4] = ft1;
                head_flag[0][t3] = 0;
            } 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        last_partial_sums[lid+GET_CONFLICT_OFFSET(lid)] = 
            last_partial_sums[lid+GET_CONFLICT_OFFSET(lid)]+ft;
    }
#endif
     //0.1627ms
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
            inter[gid] = last_partial_sums[NB_LSIZE_1+NB_LSIZE_P];
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
            ft1=0.0;
        if(gid!=0 && t1==0){
            inter[gid] = ft1 + last_partial_sums[NB_LSIZE_1+NB_LSIZE_P];
        }
        previous_workgroups_last_partial_sum = ft1;   
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ft1 = previous_workgroups_last_partial_sum*
          ((thread_res_entry_location-block_res_entry_location)==0) 
          +(lid>0?last_partial_sums[lid-1+GET_CONFLICT_OFFSET(lid-1)]:0); //0.1706ms
    t3=1;
#if NB_REG_SIZE
    for(t1=0,t2=1<<BIT_WIDTH_1;t1<NB_REG_SIZE;t1++,t2=t2>>1)
    {
        if((bit_flag_for_register&t2)==0){
            ft=ft1*t3 + intermediate_sum_r[t1];
#if defined BLOCK_HEIGHT_3
            res[thread_res_entry_location*3]=ft.s0;
            res[thread_res_entry_location*3+1]=ft.s1;
            res[thread_res_entry_location*3+2]=ft.s2;
#else
            res[thread_res_entry_location]=ft;
#endif
            t3=0;
            thread_res_entry_location++;
        }
    }
#endif
#if NB_LOCAL_SIZE
    for(t1=0,t2=1<<BIT_WIDTH_1;t1<NB_LOCAL_SIZE;t1++,t2=t2>>1)
    {
        if((bit_flag_for_localmem&t2)==0){
            ft=ft1*t3 + intermediate_sum_l[ctaid][laneid][t1];
#if defined BLOCK_HEIGHT_3
            res[thread_res_entry_location*3]=ft.s0;
            res[thread_res_entry_location*3+1]=ft.s1;
            res[thread_res_entry_location*3+2]=ft.s2;
#else
            res[thread_res_entry_location]=ft;
#endif
            t3=0;
            thread_res_entry_location++;
        }
    }  //0.1834ms
#endif
}
