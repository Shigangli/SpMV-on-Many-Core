#pragma OPENCL EXTENSION cl_khr_fp64:enable
#define LOG_NUM_BANKS 5 
#define NUM_BANKS 32 
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)

#if defined (NB_VEC_16)
#define VEC_TYPE unsigned short
#define VEC_CHAR_LEN 2
#endif
#if defined (NB_VEC_32)
#define VEC_TYPE unsigned int
#define VEC_CHAR_LEN 4
#endif
#if defined (NB_VEC_1)
#define VEC_TYPE double
#define VEC_CHAR_LEN 4
#endif
#if defined (NB_VEC_2)
#define VEC_TYPE double2
#define VEC_CHAR_LEN 8
#endif
#if defined (NB_VEC_4)
#define VEC_TYPE double4
#define VEC_CHAR_LEN 16
#endif
#if defined (NB_VEC_8)
#define VEC_TYPE double8
#define VEC_CHAR_LEN 32
#endif

__kernel void transpose(__global VEC_TYPE *gdst, __global VEC_TYPE * gsrc,int cta, int src_datalen){
    int tid = get_global_id(0);
    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int gsize = get_global_size(0);
#if defined TRANS_SIZE
    int kgp = lid / TRANS_SIZE;
    int kid = lid % TRANS_SIZE;
    int src_id=gid * cta * NB_LSIZE + kgp * TRANS_SIZE * cta + kid;
    int dst_id=tid;
    __local VEC_TYPE lm[NB_LSIZE/TRANS_SIZE][TRANS_SIZE][TRANS_SIZE+1];
    for(int i=0;i<cta / TRANS_SIZE; i++)
    {
        for(int j=0;j<TRANS_SIZE; j++){
            int t = src_id + j * cta + i * TRANS_SIZE;
            t = t < src_datalen/VEC_CHAR_LEN ? t : src_datalen/VEC_CHAR_LEN - 1;
            lm[kgp][j][kid] = gsrc[t];
        }
        for(int j=0;j<TRANS_SIZE; j++){
            int t = dst_id + (j + i * TRANS_SIZE)*gsize;
            gdst[t] = lm[kgp][kid][j];
        }
    }
#else
    int data_id = tid;
    for(int i=0;i<cta;i++){
        if(data_id+gsize*i<src_datalen/VEC_CHAR_LEN){
            gdst[data_id+gsize*i] = gsrc[data_id + gsize*i];
        }
        else{
            gdst[data_id + gsize*i] = 0;
        }
    }
#endif
}
