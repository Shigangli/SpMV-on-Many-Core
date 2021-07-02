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
#endif

#if defined (NB_L128)
#define NB_LSIZE 128
#define NB_LSIZE_LOG 7 
#define H_NB_LSIZE 64
#define NB_LSIZE_1 127
#define NB_LSIZE_2 129
#define NB_LSIZE_P 3 
#endif

#if defined (NB_L256)
#define NB_LSIZE 256 
#define NB_LSIZE_LOG 8 
#define H_NB_LSIZE 128
#define NB_LSIZE_1 255 
#define NB_LSIZE_2 257 
#define NB_LSIZE_P 7 
#endif

#if defined (NB_L512)
#define NB_LSIZE 512 
#define NB_LSIZE_LOG 9 
#define H_NB_LSIZE 256
#define NB_LSIZE_1 511 
#define NB_LSIZE_2 513 
#define NB_LSIZE_P 15
#endif
//#define DYNAMIC_TASK

#if defined (NB_VEC_1)
#define VEC_TYPE float
#define vec_sum(r) (r)
#endif
#if defined (NB_VEC_2)
#define VEC_TYPE float2
#define vec_sum(r) (r.s0+r.s1)
#endif
#if defined (NB_VEC_4)
#define VEC_TYPE float4
#define vec_sum(r) (r.s0+r.s1+r.s2+r.s3)
#endif
#if defined (NB_VEC_8)
#define VEC_TYPE float8
#define vec_sum(r) (r.s0+r.s1+r.s2+r.s3+r.s4+r.s5+r.s6+r.s7)
#endif
#if defined TX_1
#if defined NB_VEC_1
#define GET_VECTOR(rvector,vector,loc) \
    int2 coord; \
    coord.x = loc % TX_WIDTH; \
    coord.y = loc / TX_WIDTH; \
    float4 tvector = read_imagef(vector,smp,coord);\
    rvector = tvector.s0;
#endif
#if defined NB_VEC_2
#define GET_VECTOR(rvector,vector,loc) \
    int2 coord; \
    coord.x = loc % TX_WIDTH; \
    coord.y = loc / TX_WIDTH; \
    float4 tvector = read_imagef(vector,smp,coord);\
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
    float4 tvector = read_imagef(vector,smp,coord);\
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
#define SEG_TYPE float
#define RES_TYPE float
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst = dst + vec_sum(temp);
#endif
#if defined BLOCK_HEIGHT_2
#define SEG_TYPE float2
#define RES_TYPE float2
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst.s0 = dst.s0 + vec_sum(temp);\
    temp = data1[data_id];\
    temp = temp * rvector;\
    dst.s1 = dst.s1 + vec_sum(temp);
#endif
#if defined BLOCK_HEIGHT_3
#define SEG_TYPE float4
#define RES_TYPE float
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst.s0 = dst.s0 + vec_sum(temp);\
    temp = data1[data_id];\
    temp = temp * rvector;\
    dst.s1 = dst.s1 + vec_sum(temp);\
    temp = data2[data_id];\
    temp = temp * rvector;\
    dst.s2 = dst.s2 + vec_sum(temp);
#endif
#if defined BLOCK_HEIGHT_4
#define SEG_TYPE float4
#define RES_TYPE float4
#define GET_DATA(dst) \
    temp = data[data_id];\
    temp = temp * rvector;\
    dst.s0 = dst.s0 + vec_sum(temp);\
    temp = data1[data_id];\
    temp = temp * rvector;\
    dst.s1 = dst.s1 + vec_sum(temp);\
    temp = data2[data_id];\
    temp = temp * rvector;\
    dst.s2 = dst.s2 + vec_sum(temp);\
    temp = data3[data_id];\
    temp = temp * rvector;\
    dst.s3 = dst.s3 + vec_sum(temp);
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
                         __global int *res_entry, //result entry for each thread.
                         __global volatile SEG_TYPE *inter, // the buffer used for adjacent synchronization
                         __global int *para_scan,  //the signal to specify if the parallel segmented scan can be removed
                         int groupnum,           //the number of the launched workgroups.

#if defined TX_1
                         __read_only image2d_t vector,
#else
                         __global VEC_TYPE * __restrict vector,
#endif
                         __global RES_TYPE *res) 
{
    unsigned int lid = get_local_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local SEG_TYPE res_cache[CACHE_LEN], last_partial_sums[NB_LSIZE+NB_LSIZE_P];
    __local unsigned int head_flag[2][NB_LSIZE+NB_LSIZE_P];  //head flag array for parallel segmented scan
    __local unsigned int block_res_entry_location; //the first result localtion of the current workgroup
    __local unsigned int block_res_end_location; //the first result localtion of the next workgroup
    SEG_TYPE sum=0.0,ft=0.0;
    VEC_TYPE temp,rvector;
    unsigned int gid = get_group_id(0);
    unsigned int tid = gid * NB_LSIZE + lid, data_id = tid;
    unsigned int t1=0,t2=0,t3=0,t4=col[data_id];
    unsigned int thread_res_entry_location,thread_res_entry_location_bak;
    unsigned int block_res_end_location_2,block_res_entry_location_2;
    unsigned BIT_TYPE bit_flag,accumulate_bit_flag=I_MAX;
    thread_res_entry_location = res_entry[tid];
    thread_res_entry_location_bak = thread_res_entry_location;
    if(lid==0) {
        block_res_entry_location = thread_res_entry_location;
        block_res_end_location = res_entry[tid + NB_LSIZE];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    block_res_entry_location_2 = block_res_entry_location;
    block_res_end_location_2 = block_res_end_location;
    t3 = groupnum * NB_LSIZE;   //0.0128ms
    for(int i=0;i<NB_REG_GRP;i++){
        bit_flag=bit[tid*NB_REG_GRP+i];
        accumulate_bit_flag = bit_flag & accumulate_bit_flag; 
        for(t1=0,t2=1<<BIT_WIDTH_1;t1<BIT_WIDTH;t1++,t2=t2>>1){
            {GET_VECTOR(rvector,vector,t4)}  //0.1002ms
            GET_DATA(sum)
            if((bit_flag&t2)==0){
                if(thread_res_entry_location-block_res_entry_location_2<CACHE_LEN){
                    res_cache[thread_res_entry_location-block_res_entry_location_2]=sum;  //0.1230ms
                }
                else{
#if defined BLOCK_HEIGHT_3
                    res[thread_res_entry_location*3] = sum.s0;
                    res[thread_res_entry_location*3+1] = sum.s1;
                    res[thread_res_entry_location*3+2] = sum.s2;
#else
                    res[thread_res_entry_location] = sum;
#endif
                }
                sum=0.0;
                thread_res_entry_location++;
            }
            data_id += t3;
            if( data_id < groupnum * NB_REG_GRP * BIT_WIDTH *NB_LSIZE ){
                GET_COL(col,data_id,t4)
            }
        }  //0.1237ms
    }
    last_partial_sums[lid+GET_CONFLICT_OFFSET(lid)] = sum;
#if defined SEGSCAN_1 
    if(para_scan[gid]==1){     //parallel segmented scan on the last partial sums.
        head_flag[0][lid+GET_CONFLICT_OFFSET(lid)]=1 - (accumulate_bit_flag == I_MAX);  //generate the head flags
        head_flag[1][lid+GET_CONFLICT_OFFSET(lid)]=1 - (accumulate_bit_flag == I_MAX);  //backup the head flags
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
        for(t2 =1; t2<NB_LSIZE;t2<<=1){
            barrier(CLK_LOCAL_MEM_FENCE);
            t1>>=1; 
            t3 = t1 * ((lid<<1)+1)-1,t4 = t3 + t1;
            t3 += GET_CONFLICT_OFFSET(t3);
            t4 += GET_CONFLICT_OFFSET(t4);
            if(lid < t2){
                ft = last_partial_sums[t3];
                last_partial_sums[t3] = last_partial_sums[t4];
	        if(head_flag[1][t1 * ((lid<<1)+1)+GET_CONFLICT_OFFSET(t1 * ((lid<<1)+1))] == 1){
                    ft = 0.0;
                }
                if(head_flag[1][t1 * ((lid<<1)+1)+GET_CONFLICT_OFFSET(t1 * ((lid<<1)+1))] != 1){
                    if(head_flag[0][t3]!=1)
                        ft = last_partial_sums[t4]+ft;
                }
                last_partial_sums[t4] = ft;
                head_flag[0][t3] = 0;
            } 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        last_partial_sums[lid+GET_CONFLICT_OFFSET(lid)] = last_partial_sums[lid+GET_CONFLICT_OFFSET(lid)]+sum;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid==0){
        t1 = block_res_end_location_2 - block_res_entry_location_2;
        if(gid ==0||t1!=0){
            inter[gid] = last_partial_sums[NB_LSIZE_1+NB_LSIZE_P];
        }
        if(gid != 0){
#if defined BLOCK_HEIGHT_1
            while((ft=inter[gid-1])==FLT_MAX){}
#else
            while(1){
                ft=inter[gid-1];
                if(ft.s0!=FLT_MAX)
                  break;
            }
#endif
            inter[gid-1]=FLT_MAX;
        }
        else
            ft=0.0;
        if(gid!=0 && t1==0){
            inter[gid] = ft + last_partial_sums[NB_LSIZE_1+NB_LSIZE_P];
        }
    }
    if(thread_res_entry_location_bak-block_res_entry_location_2<CACHE_LEN&&
        lid>0&&(accumulate_bit_flag!=I_MAX)){
        res_cache[thread_res_entry_location_bak-block_res_entry_location_2] =
        res_cache[thread_res_entry_location_bak-block_res_entry_location_2] + 
        last_partial_sums[lid-1+GET_CONFLICT_OFFSET(lid-1)]; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(thread_res_entry_location_bak-block_res_entry_location_2>=CACHE_LEN&&    //if overflowed
        lid>0&&(accumulate_bit_flag!=I_MAX)){
        ft = last_partial_sums[lid-1+GET_CONFLICT_OFFSET(lid-1)]; 
#if defined BLOCK_HEIGHT_3
        res[thread_res_entry_location_bak*3] = res[thread_res_entry_location_bak*3]+ft.s0;
        res[thread_res_entry_location_bak*3+1] = res[thread_res_entry_location_bak*3+1]+ft.s1;
        res[thread_res_entry_location_bak*3+2] = res[thread_res_entry_location_bak*3+2]+ft.s2;
#else
        res[thread_res_entry_location_bak] = res[thread_res_entry_location_bak]+ft;
#endif
    }
    for(t1=lid;t1<CACHE_LEN;t1+=NB_LSIZE)   //flush the result in the cache into global memory.
    {
        if(block_res_entry_location_2+t1<block_res_end_location_2)
        {
#if defined BLOCK_HEIGHT_3
            res[(block_res_entry_location_2+t1)*3] =   res_cache[t1].s0+(t1==0?ft.s0:0);
            res[(block_res_entry_location_2+t1)*3+1] = res_cache[t1].s1+(t1==0?ft.s1:0);
            res[(block_res_entry_location_2+t1)*3+2] = res_cache[t1].s2+(t1==0?ft.s2:0);
#else
            res[block_res_entry_location_2+t1] = res_cache[t1]+(t1==0?ft:0);
#endif
        }
    }
}
